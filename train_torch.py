import torch
import torch.nn as nn
import yaml
from utils import get_class, ScalarTransforms
from replay_buffer import ReplayBuffer, ObservationTrajectory
from tqdm import tqdm
from statistics import mean
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter 
import os
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Ensures deterministic behavior (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42) 

def loss_fn(
        observed_reward: torch.tensor, 
        predicted_reward: torch.tensor, 
        bootstrapped_reward: torch.tensor, 
        predicted_value: torch.tensor, 
        visit_counts: torch.tensor, 
        predicted_policy: torch.tensor, 
        target_transformation, #_supports_representation
        K: int
    ):
    """
    Args:
        predicted_reward: predicted distribution over supports
        predicted_value: predicted distributions over supports
        target_transformation: transforms the target scalar to a supports representations and extracts the coefficient vector for the supports
    """
    reward_loss = F.kl_div( 
        F.log_softmax(predicted_reward.view(-1, predicted_reward.shape[-1]), dim=-1), 
        target_transformation(observed_reward).view(-1, predicted_reward.shape[-1]),
        reduction="batchmean"
    )
    value_loss = F.kl_div(
        F.log_softmax(predicted_value.view(-1, predicted_value.shape[-1]), dim=-1), 
        target_transformation(bootstrapped_reward).view(-1, predicted_value.shape[-1]),
        reduction="batchmean"
    )

    visit_counts_normalized = visit_counts / visit_counts.sum(dim=-1, keepdim=True)
    policy_loss = F.kl_div(
        F.log_softmax(predicted_policy.view(-1, predicted_policy.shape[-1]), dim=-1),
        visit_counts_normalized.view(-1, visit_counts_normalized.shape[-1]),
        reduction="batchmean"
    )
    return (1/K) * (reward_loss + value_loss + policy_loss), reward_loss, value_loss, policy_loss


class RLSystem:
    def __init__(self, cfg: dict):
        #hyperparams
        self.num_episodes = cfg["num_episodes"]
        self.num_steps = cfg["num_steps"] 
        self.num_simulations = cfg["num_simulations"]
        self.minibatch_size = cfg["minibatch_size"]
        self.real_resolution = cfg["real_resolution"]
        self.latent_resolution = cfg["latent_resolution"]
        self.state_history_length = cfg["model"]["state_history_length"]
        self.actions = cfg["actions"]
        self.n_actions = len(self.actions)
        self.K = cfg["num_unroll_steps"]
        self.gamma = cfg["discount_factor"]
        self.iterations = cfg["num_iterations"]
        self.epochs = cfg["epochs"]
        self.n_parallel = cfg["n_parallel"]
        self.temperature = 1.0
        self.max_steps_test = 200

        #internal classes
        mu_zero_class = get_class("src.networks", cfg["model"]["agent_name"])
        self.mu_zero = mu_zero_class(cfg["model"]) 
        self.mu_zero_target = mu_zero_class(cfg["model"]) #Target Network
        self.mu_zero_target.load_state_dict(self.mu_zero.state_dict())
        # latent_mcts_class = get_class("src.parallel_mcts", cfg["search"]["mcts_name"])
        latent_mcts_class = get_class("src.parallel_mcts_2", cfg["search"]["mcts_name"])
        self.scalar_transforms = ScalarTransforms(cfg["model"])
        self.latent_mcts = latent_mcts_class(cfg, self.mu_zero_target, self.scalar_transforms) #NOTE: passing the networks like this is not good
        environment_class = get_class(cfg["environment"]["environment_path"], cfg["environment"]["environment_name"])
        self.environment = environment_class(cfg["environment"]) 

        #Disable gradients for Target Network
        for param in self.mu_zero_target.parameters():
            param.requires_grad = False

        #Replay Buffer
        self.replay_buffer = ReplayBuffer(self.state_history_length, self.K, cfg["replay_buffer_max"], cfg["discount_factor"], self.n_parallel)
        self.observation_trajectories = []
        self.samples_before_train = cfg["samples_before_train"]
        self.training_iteration = 0
        self.training_step = 0
        self.num_batches = cfg["num_batches"]
        # self.acting_steps = 0 

        #logging 
        self.logdir = "logs/train_data/"
        self.filewriter = SummaryWriter(self.logdir)
        self.acting_step = 0

        #statistics
        self.action_stats = torch.tensor([])

        self.init_iteration = 0
        self.load_weights = cfg["load_weights"]
        self.checkpoint_path = cfg["checkpoint_path"]
        if self.load_weights:
            self._load_weights()

        self.mcts_iter = 0
        self.saved_weights = 0

    def train(self):
        """
        The training loop iteratively exchanges between acting stage and training stage
        """
        started_training = False
        for iteration in range(self.init_iteration, 50000):
            if self.training_iteration > 10:
                #start temperature decay
                self.temperature *= 0.996
                self.temperature = max(self.temperature, 0.1)
                
            if self.training_iteration == 100:
                self.latent_mcts.noise_weight = 0.1

            if iteration % 15 == 0 and iteration != 0 and started_training:
                self.load_latest_weights() #update Target Network
                print(f"Updates Target Network. STEPS: {self.training_step}")

            #episode generation
            print(f"ACTING STAGE {iteration}")
            self._acting_stage()

            #network training
            print(f"TRAINING STAGE {iteration}")
            if self.replay_buffer.length > self.samples_before_train or 1:
                self._training_stage()
                self.training_iteration += 1
                started_training = True

            if iteration % 15 == 0 and self.replay_buffer.length > self.samples_before_train:
                self._save_weights(iteration)
                print("SAVED WEIGHTS!")

            print(f"REPLAY_BUFFER: {self.replay_buffer.length}, STEPS: {self.training_step}")

        self._save_weights(iteration)

    def _acting_stage(self):
        """
        Sequential
        """
        self.mu_zero_target.eval_mode()
        for episode in tqdm(range(self.num_episodes)):
            initial_state, done = self.environment.reset()
            self._pad_initial_state(self.convert_to_grayscale(initial_state)) #resets self.observation_trajectories

            self._run_episode(initial_state)

    def _run_episode(self, state: torch.tensor):
        """
        Runs an entire episode in the environment, from start to finish - Plays the game

        Args:
            state (batch_size, 3, 16, 16)
        """        
        done_mask = torch.zeros((state.shape[0]), dtype=torch.bool)
        prev_done_mask = done_mask
        valid_actions = torch.ones((state.shape[0], self.n_actions)) 
        warp_state = self.convert_to_grayscale(state)

        length_counter = 0
        while not torch.all(done_mask == True):

            if length_counter > 260:
                break
            
            value, visit_counts = self._sample_action(warp_state, valid_actions) #v, π

            #temperature based sampling
            visit_counts_temp = visit_counts ** (1/self.temperature)
            probs = visit_counts_temp / visit_counts_temp.sum(dim=1, keepdim=True)
            action = torch.zeros(probs.shape[0], dtype=torch.long)

            for i in range(probs.shape[0]):
                dist = torch.distributions.Categorical(probs[i])
                action[i] = dist.sample()


            state, reward, done_mask, valid_actions = self.environment.step(state, action, done_mask)

            warp_state = self.convert_to_grayscale(state)
            for idx in range(len(self.observation_trajectories)):
                if not prev_done_mask[idx]: #game is not finished
                    self.observation_trajectories[idx].add_observation(
                        action[idx], warp_state[idx], reward[idx], visit_counts[idx], value[idx] #value and visit_counts are for previous state
                    )
            prev_done_mask = done_mask.clone()

            length_counter += 1
            self.action_stats = torch.cat((self.action_stats, action), dim=0)

        #episode statistics
        episode_lens = [obs.length for obs in self.observation_trajectories]
        max_len = max(episode_lens)
        avg_len = mean(episode_lens) 
        value_counts = torch.bincount(self.action_stats.long(), minlength=self.n_actions)
        print(f"Max episode length: {max_len}, Average episode length: {avg_len}")
        print(f"Value counts 0: {value_counts[0]}, 1: {value_counts[1]}, 2: {value_counts[2]}")

        #save to replay buffer
        for observation_trajectory in self.observation_trajectories:
            if observation_trajectory.length > (self.K + 1):
                self.replay_buffer.save_observation_trajectory(observation_trajectory)

        del self.action_stats
        self.action_stats = torch.tensor([])
        
        rewards = self.replay_buffer.get_reward_sums()
        avg_reward = mean(rewards)
        self.saved_weights += 1
        self.filewriter.add_scalar("Reward/avg", avg_reward, global_step=self.acting_step)
        self.acting_step += 1


    def _sample_action(self, state: torch.tensor, mask: list): 
        """
        Latent-MCTS + UCB
        Here we operate in the latent space, via the MCTS algorithm

        Args:
            state (batch_size, 3, 16, 16)
            mask (32, 3): 0/1 mask representing if action is legal or not

        Returns:
            value (batch_size): 
            visit_counts (batch_size, n_actions)
        """
        repnet_inputs = []
        for idx, observation_trajectory in enumerate(self.observation_trajectories):
            repnet_input = self._prepare_mcts_input(state[idx], observation_trajectory, self.real_resolution,) 
            repnet_inputs.append(repnet_input) 
        repnet_inputs = torch.stack(repnet_inputs)
        hidden_state = self.mu_zero_target.create_hidden_state_root(repnet_inputs) #
        
        value, visit_counts = self.latent_mcts.search(hidden_state, mask, self.training_iteration)
        return value, visit_counts

    def _prepare_mcts_input(self, state: torch.tensor, observation_trajectory: ObservationTrajectory, resolution: tuple):
        """
        Prepares a tensorized input from the sequence of states and actions

        Args:
            state (channels, resolution[0], resolution[1]): the current state

        Returns:
            repnet_input (1, 128, resolution[0], resolution[1]): encoded representation of the last 32 (32*3) states concatenated with last 32 actions 
        """
        #state = (batch, channels, res[0], res[1])
        actions = observation_trajectory.get_actions()[-self.state_history_length:]
        action_plane = self._encode_actions(actions.unsqueeze(0), self.state_history_length, resolution).squeeze(0) #(1, history_len, 16, 16)

        state_sequence = observation_trajectory.get_states()[-(self.state_history_length-1):].view(-1, resolution[0], resolution[1]) #(num_states*channels, res[0], res[1])
        state_sequence = torch.cat((state_sequence, state), dim=0)

        repnet_input = torch.cat((state_sequence, action_plane), dim=0)
        return repnet_input
    
    def _encode_actions(self, actions: torch.tensor, n_actions: int, resolution: tuple):
        """
        Takes in a list of actions and returns a tensor of shape (self.state_history_length, resolution[0], resolution[1])

        Args:
            actions (bs, actions): the actions we want to encode (create bias plane)
            n_actions (1,): the number of actions we want to create bias plane encoding for 
            resolution (2): tuple of resolution for the bias plane

        Returns:
            bias_plane (bs, n_actions, resolution[0], resolution[1])
        """
        actions_expanded = (actions / self.n_actions)[:, :, None, None].expand(-1, -1, resolution[0], resolution[1]) 
        bias_plane = torch.ones((actions.shape[0], n_actions, resolution[0], resolution[1]), device=actions_expanded.device) *  actions_expanded
        return bias_plane
    
    def _encode_action_dynamics(self, action: torch.Tensor, resolution: tuple, n_actions: int):
        """
        Encodes a batch of actions into a tensor with n_actions planes, as per MuZero.

        Args:
            action (torch.Tensor): Tensor of shape (batch,) containing chosen actions (integers).
            resolution (tuple): (height, width) of the output encoding.
            n_actions (int): Total number of possible actions.

        Returns:
            action_encoded (batch, n_actions, resolution[0], resolution[1]): One-hot encoded action planes.
        """
        batch_size = action.shape[0]
        action_encoded = torch.zeros(batch_size, n_actions, resolution[0], resolution[1], device=action.device)
        action_one_hot = F.one_hot(action, num_classes=n_actions).float()
        action_encoded = action_one_hot.view(batch_size, n_actions, 1, 1).expand(-1, -1, resolution[0], resolution[1]) #tile across spatial dimensions
        return action_encoded

    def _pad_initial_state(self, initial_state):
        """
        Generates static observation trajectories for the self.state_history_length first states
        'states' only has self.state_history_length-1 tensors because the last state is added as a result of a real transition

        Args:
            initial_state: The randomly initialized state that will be repeated (self.state_history_length-1) times
        """
        self.observation_trajectories = []
        for idx in range(self.n_parallel):    
            observation_trajectory = ObservationTrajectory(
                actions=[0 for _ in range(self.state_history_length)],
                states=[initial_state[idx] for _ in range(self.state_history_length - 1)],
                rewards=[0 for _ in range(self.state_history_length)],
                visit_counts=[torch.zeros(self.n_actions) for _ in range(self.state_history_length)],
                values=[0.0 for _ in range(self.state_history_length)],
                length=0,
                reward_sum=0
            )
            self.observation_trajectories.append(observation_trajectory)

    def convert_to_grayscale(self, state: torch.Tensor) -> torch.Tensor:
        """
        Converts a (batch, 3, height, width) breakout state to a (batch, 1, height, width) grayscale version.

        Args:
            state (torch.Tensor): The original 3-channel state tensor.

        Returns:
            torch.Tensor: Grayscale tensor with 1 channel per state.
        """
        # Define grayscale intensities for each channel
        paddle_intensity = 0.3
        ball_intensity = 1.0
        brick_intensity = 0.6

        # Split channels
        paddle = state[:, 0] * paddle_intensity
        ball   = state[:, 1] * ball_intensity
        bricks = state[:, 2] * brick_intensity

        # Combine to single channel
        grayscale = paddle + ball + bricks
        grayscale = grayscale.clamp(0, 1).unsqueeze(1)  # Add channel dim back: (batch, 1, height, width)

        return grayscale


    def load_latest_weights(self):
        """
        Loads the latest model weights from the learner (self.mu_zero)
        """
        self.mu_zero_target.load_state_dict(
            self.mu_zero.state_dict()
        )

    def _training_stage(self):
        """
        Training loop 
        """
        self.mu_zero.train_mode()

        losses = []

        sampling_idxs = torch.randperm(int(self.replay_buffer.length))
        for batch_start in tqdm(range(0, self.num_batches * self.minibatch_size, self.minibatch_size)):
            """
            k_step_policies: observed visist-counts used to compare with policy_prediction at each step k in the k-step rollout
            k_step_actions: selected actions from sample trajectory. Used for making hidden state transitions in the k-step rollout
            k_step_rewards: observed rewards used to compare with predicted rewards from the k-step rollout
            k_step_value: observed value from MCTS search used to compare with predicted value at each step k
            """
            self.mu_zero.optimizer.zero_grad()

            #prepare data from Replay Buffer
            with torch.no_grad():
                batch_end = batch_start + self.minibatch_size
                batch_idxs = sampling_idxs[batch_start:batch_end]
                mbs_input_actions, mbs_input_states, k_step_policies, k_step_actions, k_step_rewards, k_step_value = self._prepare_minibatch(batch_idxs, device)
                input_actions_encoded = self._encode_actions(mbs_input_actions, self.state_history_length, self.real_resolution) #(batch, history_len, 16, 16)
            
            #forward pass
            predicted_reward, predicted_value, predicted_policy = self._k_step_rollout(mbs_input_states, input_actions_encoded, k_step_actions) #(batch, K, 601), ..., (batch, K, n_actions)
            
            #backward
            (loss, reward_loss, value_loss, policy_loss) = loss_fn(
                observed_reward=k_step_rewards,
                predicted_reward=predicted_reward,
                bootstrapped_reward=k_step_value, #bootstrapped_rewards, 
                predicted_value=predicted_value,
                visit_counts=k_step_policies, 
                predicted_policy=predicted_policy,
                target_transformation=self.scalar_transforms.supports_representation,
                K=self.K
            )

            losses.append(loss.item())
            loss.backward()
            self.mu_zero.optimizer.step()
            self.training_step += 1

        # Compute metrics
        avg_loss = mean(losses)
        avg_reward = mean(self.replay_buffer.get_reward_sums())

        # TensorBoard logging: Loss and Reward
        global_step = self.training_iteration 
        self.filewriter.add_scalar("Loss/train", avg_loss, global_step=global_step)
        self.filewriter.add_scalar("Loss/reward", reward_loss, global_step=global_step)
        self.filewriter.add_scalar("Loss/value", value_loss, global_step=global_step)
        self.filewriter.add_scalar("Loss/policy", policy_loss, global_step=global_step)
        
        if self.training_step == 0:
            """
            Visualize the input state transitions for the training samples
            """
        
            state_seq_0 = mbs_input_states[0].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255
            state_seq_1 = mbs_input_states[1].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255
            state_seq_2 = mbs_input_states[2].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255      
            state_seq_3 = mbs_input_states[5].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255      
            state_seq_4 = mbs_input_states[6].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255      

            for i in range(self.state_history_length):
                # Add each frame as a separate step in the same run
                image = torch.cat((state_seq_0[i], state_seq_1[i], state_seq_2[i], state_seq_3[i], state_seq_4[i]), dim=-1)
                self.filewriter.add_image(f"trajectory_{self.training_iteration}_{self.training_step}/frame", image, global_step=i, dataformats="CHW")

            action_sequence = mbs_input_actions[0]
            action_log = "\n".join([f"Timestep {t}: Action {int(action_sequence[t])}" for t in range(self.state_history_length)])
            self.filewriter.add_text("trajectory_actions_" + str(self.training_step), action_log)
        print(f"Epoch {self.training_step} loss: {avg_loss}") # mean-reward: {avg_reward}")

        
        #run test simulation with the new weights
        self.environment.batch = 2
        self.latent_mcts.mu_zero = self.mu_zero #make sure the latent representations are created with the latest weights
        self.run_test_simulation(batch=2)
        self.latent_mcts.mu_zero = self.mu_zero_target #reset to use target network
        self.environment.batch = self.n_parallel

    def _prepare_minibatch(self, batch_idxs: torch.tensor, device: str):
        """ 
        Needs to:
            Sample (uniformly) a current state and the previous 32 states + actions
            Sample next K rewards and value-counts (policy distributions)

        Returns:
            action_history_minibatch (batch, len_history): action sequence up to current (root) state
            state_history_minibatch (batch, 3*len_history, 16, 16): state sequence up to current state 
            k_step_policies (batch, K, n_actions): K MCTS visit-counts statistics after current state 
            k_step_actions (batch, K): K actions taken after current state
            k_step_rewards (batch, K): K rewards gained after current state
            k_step_values (batch, K): estimated values for K states after current
        """
        #used for RepNet input
        action_history_minibatch = self.replay_buffer.get_batched_past_actions(batch_idxs)
        state_history_minibatch = self.replay_buffer.get_batched_states(batch_idxs).view(self.minibatch_size, -1, self.real_resolution[0], self.real_resolution[1])
        
        #used for training loss
        k_step_policies = self.replay_buffer.get_batched_visit_counts(batch_idxs)
        k_step_actions = self.replay_buffer.get_batched_future_actions(batch_idxs) #NOTE: Tends to be mostly just 1s in the start (biased)
        k_step_rewards = self.replay_buffer.get_batched_rewards(batch_idxs)
        k_step_values = self.replay_buffer.get_batched_values(batch_idxs) 

        return (
            action_history_minibatch.to(device),
            state_history_minibatch.to(device),
            k_step_policies.to(device),
            k_step_actions.to(device),
            k_step_rewards.to(device),
            k_step_values.to(device)
        )

    def _k_step_rollout(self, input_states: torch.tensor, input_actions: torch.tensor, k_step_actions: torch.tensor):
        """
        Recursively creates hidden states by calling Dynamics function K times, and evaluates each hidden state with Prediction function
        
        Args:
            input_states (batch_size, state_history_length * 3, resolution[0], resolution[1]): 
            input_actions (batch_size, state_history_length, resolution[0], resolution[1]): 
            k_step_actions (batch_size, K)      (batch_size, K, resolution[0], resolution[1]): 

        Returns:
            predicted_reward (batch_size, K, 601)
            predicted_value (batch_size, K, 601)
            predicted_policy (batch_size, K, n_actions)
        """
        #RepNetwork transition
        repnet_input = torch.cat((input_states, input_actions), dim=1)
        hidden_state = self.mu_zero.create_hidden_state_root(repnet_input)

        policy_stack = []
        value_stack = []
        reward_stack = []
        for k in range(self.K):
            #PredNet evaluation
            """
                policy_distribution: (bs, n_actions)
                value: (bs, 601)
            """
            policy_distribution, value = self.mu_zero.evaluate_state(hidden_state)
            policy_stack.append(policy_distribution)
            value_stack.append(value)

            #DynNet transition
            """
                hidden_state: (bs, ?, res, res)
                reward: (bs, 601)
            """
            actions_encoded = self._encode_action_dynamics(k_step_actions[:, k], self.latent_resolution, self.n_actions)
            hidden_state, reward = self.mu_zero.hidden_state_transition(hidden_state, actions_encoded) #need proper action encoding here
            reward_stack.append(reward)

        #save shit
        return torch.stack(reward_stack, dim=1), torch.stack(value_stack, dim=1), torch.stack(policy_stack, dim=1)
    
    def run_test_simulation(self, batch):
        """
        Runs a game with the new model weights, logging it to tensorboad

        Args:
            batch (int): number of simulations to run in parallel
        """
        self.mu_zero.eval_mode()

        #prepare inputs
        obs_trajectories = []
        state, _ = self.environment.reset()[:batch]
        warp_state = self.convert_to_grayscale(state)
        valid_actions = torch.ones((batch, self.n_actions))
        done = torch.tensor([False for _ in range(batch)])
        for idx in range(batch):
            observation_trajectory = ObservationTrajectory(
                actions=[1 for _ in range(self.state_history_length)],
                states=[warp_state[idx] for _ in range(self.state_history_length - 1)],
                rewards=[0 for _ in range(self.state_history_length)],
                visit_counts=[torch.zeros(self.n_actions) for _ in range(self.state_history_length)],
                values=[0.0 for _ in range(self.state_history_length)],
                length=0,
                reward_sum=0
            )
            obs_trajectories.append(observation_trajectory)

        #perform search
        frames = [[] for _ in range(batch)]
        step_i = 0
        with torch.no_grad():
            while not torch.all(done == True):
                if step_i > self.max_steps_test:
                    break
                #sample action
                repnet_inputs = []
                for i in range(batch):
                    repnet_input = self._prepare_mcts_input(warp_state[i], obs_trajectories[i], self.real_resolution)
                    repnet_inputs.append(repnet_input)
                hidden_state = self.mu_zero.create_hidden_state_root(torch.stack(repnet_inputs))
                value, visit_counts = self.latent_mcts.search(hidden_state, valid_actions, self.training_iteration)

                #temperature based sampling
                temperature = 0.1
                visit_counts_temp = visit_counts ** (1/temperature)
                probs = visit_counts_temp / visit_counts_temp.sum(dim=1, keepdim=True)
                action = torch.zeros(probs.shape[0], dtype=torch.long)

                for i in range(probs.shape[0]):
                    dist = torch.distributions.Categorical(probs[i])
                    action[i] = dist.sample()

                #step in environment
                state, reward, done, valid_actions = self.environment.step(state, action, done)
                warp_state = self.convert_to_grayscale(state)
                
                #write the frame
                for idx in range(batch):
                    if not done[idx]:
                        frames[idx].append(warp_state[idx])
                        
                if step_i % 10 == 0:
                    print(f"STEP: {step_i}")
                step_i += 1

                #save to observation trajectory
                for i in range(batch):
                    obs_trajectories[i].add_observation(
                        action[0].item(), warp_state[i], reward[i], visit_counts[i], value[i]
                    )

        #write the max idx to tensorboard
        #NOTE: change to iterate over all frames if it is desired to view all test runs
        sample = 0
        for step, frame in enumerate(frames[sample]):
            self.filewriter.add_image(f"TEST_{sample}/frame", frame, global_step=step, dataformats="CHW")

        #cleanup
        del obs_trajectories, state, hidden_state, repnet_input, value, visit_counts

        print("DONE")

    def _save_weights(self, iteration, name=None):
        """
        Saves the model weights, optimizer state, and training iteration.
        """
        if name:
            save_path = f"weights/checkpt_inf_{name}.pth" 
        else:
            save_path = "weights/checkptX2.pth"
        torch.save({
            "model_state_dict": self.mu_zero.state_dict(),
            "optimizer_state_dict": self.mu_zero.optimizer.state_dict(),  # Save optimizer state
            "training_iteration": self.training_iteration,
            "acting_step": self.acting_step,
            "iteration": iteration,            
            "replay_buffer": {
                "past_actions_buffer": self.replay_buffer.past_actions_buffer,
                "future_actions_buffer": self.replay_buffer.future_actions_buffer,
                "state_buffer": self.replay_buffer.state_buffer,
                "reward_buffer": self.replay_buffer.reward_buffer,
                "visit_counts_buffer": self.replay_buffer.visit_counts_buffer,
                "value_buffer": self.replay_buffer.value_buffer,
                "reward_sums": self.replay_buffer.reward_sums,
                "length": self.replay_buffer.length,
                "max_length": self.replay_buffer.max_length,
                "bootstrapped_values": self.replay_buffer.bootstrapped_values
            }
        }, save_path)
        print(f"Model weights, optimizer state, replay buffer and training iteration {self.training_iteration} saved to {save_path}")

    def _load_weights(self):
        """
        Loads model weights, optimizer state, and training iteration.
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.mu_zero.load_state_dict(checkpoint["model_state_dict"])
            self.mu_zero.to(self.mu_zero.device)  # Ensure model is on the correct device
            
            #restore optimizer state
            if "optimizer_state_dict" in checkpoint:
                self.mu_zero.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Optimizer state restored.")
            else:
                print("No optimizer state found in checkpoint.")

            #restore replay buffer
            replay_data = checkpoint["replay_buffer"]
            self.replay_buffer.past_actions_buffer = replay_data["past_actions_buffer"]
            self.replay_buffer.future_actions_buffer = replay_data["future_actions_buffer"]
            self.replay_buffer.state_buffer = replay_data["state_buffer"]
            self.replay_buffer.reward_buffer = replay_data["reward_buffer"]
            self.replay_buffer.visit_counts_buffer = replay_data["visit_counts_buffer"]
            self.replay_buffer.value_buffer = replay_data["value_buffer"]
            self.replay_buffer.reward_sums = replay_data["reward_sums"]
            self.replay_buffer.length = replay_data["length"]
            self.replay_buffer.max_length = replay_data["max_length"]
            self.replay_buffer.bootstrapped_values = replay_data["bootstrapped_values"]

            self.training_iteration = checkpoint.get("training_iteration", 0)  #restore training iteration
            self.acting_step = checkpoint.get("acting_step", 0)  #restore acting step
            self.init_iteration = checkpoint.get("iteration", 0)  #restore initial iteration
            print(f"Loaded model weights and training iteration {self.training_iteration} from {self.checkpoint_path}")
        else:
            print(f"No checkpoint found at {self.checkpoint_path}, training from scratch.")
    

if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    trainer = RLSystem(config["parameters"])
    trainer.train()


