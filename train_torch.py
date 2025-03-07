import torch
import torch.nn as nn
import yaml
from utils import get_class, ReplayBuffer, ObservationTrajectory
from tqdm import tqdm
from statistics import mean
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter 


reward_loss_fn = nn.NLLLoss() #used instead of CrossentropyLoss because the model outputs softmax probs
value_loss_fn = nn.NLLLoss # nn.NLLLoss(log(softmax(logits)), ...) == nn.CrossEntropyLoss(logits, ...)
policy_loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(
        observed_reward: torch.tensor, #scalars
        predicted_reward: torch.tensor, #dists
        bootstrapped_reward: torch.tensor, #scalars
        predicted_value: torch.tensor, #dists
        value_counts: torch.tensor, #scalars
        predicted_policy: torch.tensor, #dists
        target_transformation, #func
        K: int
    ):
    """
    Args:
        predicted_reward: softmax distribution over supports (remember this is what the network learns to predict, so we dont have to transform it - only invert for inference expectation!)
        predicted_value: softmax distributions over supports
        target_transformation: transforms the target scalar to a supports representations and extracts the coefficient vector for the supports
    # """
    # reward_loss = reward_loss_fn(torch.log(predicted_reward).view(-1, 601), target_transformation(observed_reward).view(-1, 601)).mean() #TODO: messy
    # value_loss = value_loss_fn(torch.log(predicted_value).view(-1, 601), target_transformation(bootstrapped_reward).view(-1, 601)).mean()
    reward_loss = torch.sum(-target_transformation(observed_reward) * torch.log(predicted_reward) + 1e-8, dim=-1).mean()
    value_loss = torch.sum(-target_transformation(bootstrapped_reward) * torch.log(predicted_value) + 1e-8, dim=-1).mean()
    policy_loss = policy_loss_fn(predicted_policy.view(predicted_policy.shape[0], -1), value_counts.view(value_counts.shape[0], -1)).mean()
    return (1/K) * (reward_loss + value_loss + policy_loss) #TODO: Add regularizing term

class RLSystem:
    def __init__(self, cfg: dict):
        #hyperparams
        self.num_episodes = cfg["num_episodes"]
        self.num_steps = cfg["num_steps"] #??
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

        #internal classes
        mu_zero_class = get_class("src.networks", cfg["model"]["agent_name"])
        self.mu_zero = mu_zero_class(cfg["model"]) 
        latent_mcts_class = get_class("src.parallel_mcts", cfg["search"]["mcts_name"])
        self.latent_mcts = latent_mcts_class(cfg, self.mu_zero) #NOTE: passing the networks like this is not good
        environment_class = get_class(cfg["environment"]["environment_path"], cfg["environment"]["environment_name"])
        self.environment = environment_class(cfg["environment"]) 
            
        #Replay Buffer
        self.replay_buffer = ReplayBuffer(self.K)
        self.observation_trajectories = []

        #logging 
        self.logdir = "logs/train_data/"
        self.filewriter = SummaryWriter(self.logdir)

    def train(self):
        """
        """
        for _ in range(self.iterations):
            #episode generation
            print("ACTING STAGE")
            self._acting_stage()

            #network training
            print("TRAINING STAGE")
            self._training_stage()

            #reset replay buffer
            self.replay_buffer.empty_buffer()

    def _acting_stage(self):
        """
        Sequential
        """
        self.mu_zero.eval_mode()
        for episode in tqdm(range(self.num_episodes)):
            initial_state = self.environment.get_initial_state() / 255 #(batch, 3, 16, 16)
            self._reset_environment() #resets self.observation_trajectories

            self._run_episode(initial_state)

    def _run_episode(self, state: torch.tensor):
        """
        Runs an entire episode in the environment, from start to finish - Plays the game

        Args:
            state (3, 16, 16)
        """
        done_mask = torch.zeros((state.shape[0]), dtype=torch.bool)
        prev_done_mask = done_mask
        valid_actions = torch.ones((state.shape[0], self.n_actions)) #TODO: Could introduce some randomization here

        #At this level we are traversing the real environment
        length_counter = 0
        while not torch.all(done_mask == True):
            value, visit_counts = self._sample_action(state, valid_actions) #v, π
            action = torch.argmax(visit_counts, dim=1)
            state, reward, done_mask, valid_actions = self.environment.step(state * 255, action, done_mask)
            state = state / 255 #normalize
            for idx in range(len(self.observation_trajectories)):
                if not prev_done_mask[idx]: #game is not finished
                    self.observation_trajectories[idx].add_observation(
                        action[idx], state[idx], reward[idx], visit_counts[idx], value[idx]
                    )
            prev_done_mask = done_mask.clone()

            """
            CRUICAL:
                Certain games get the ball behind the brick wall very early and this makes them last MUCH longer (too long)s
            """
            if length_counter > 102:
                break
            length_counter += 1

            #NOTE: could save observation trajectories here instead based on done_mask

        print(f"Episode length: {self.observation_trajectories[0].length}, {self.observation_trajectories[1].length}, {self.observation_trajectories[2].length}")
        for observation_trajectory in self.observation_trajectories:
            if observation_trajectory.length > (self.K + 1):
                self.replay_buffer.save_observation_trajectory(observation_trajectory)


    def _sample_action(self, state: torch.tensor, mask: list): 
        """
        Latent-MCTS + UCB
        Here we operate in the latent space, via the MCTS algorithm

        Returns:
            value (1,): 
            visit_counts (n_actions,)
        """
        repnet_inputs = []
        for idx, observation_trajectory in enumerate(self.observation_trajectories):
            repnet_input = self._prepare_mcts_input(state[idx], observation_trajectory, self.real_resolution,) 
            repnet_inputs.append(repnet_input) 
        repnet_inputs = torch.stack(repnet_inputs)
        hidden_state = self.mu_zero.create_hidden_state_root(repnet_inputs) #
        
        value, visit_counts = self.latent_mcts.search(hidden_state, mask) #(batch,), (batch, n_actions)
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
            actions (bs, actions)
            resolution (2): tuple of resolution
        """
        actions_expanded = (actions / self.n_actions)[:, :, None, None].expand(-1, -1, resolution[0], resolution[1]) #NOTE
        bias_plane = torch.ones((actions.shape[0], n_actions, resolution[0], resolution[1]), device=actions_expanded.device) *  actions_expanded
        return bias_plane

    def _reset_environment(self):
        """
        Sets the first 31 states in the trajectory to zero tensors
        
        self.state_history_length - 1: because the initial state is appended afterwards
        """
        self.observation_trajectories = []
        for idx in range(self.n_parallel):    
            observation_trajectory = ObservationTrajectory(
                actions=[random.randint(0, 2) for _ in range(self.state_history_length)], #TODO: Maybe randomize
                states=[torch.zeros(3, self.real_resolution[0], self.real_resolution[1]) for _ in range(self.state_history_length - 1)], #one less because we will append initial_state from env
                rewards=[0 for _ in range(self.state_history_length)],
                visit_counts=[torch.zeros(self.n_actions) for _ in range(self.state_history_length)],
                values=[0.0 for _ in range(self.state_history_length)],
                length=0,
                reward_sum=0
            )
            self.observation_trajectories.append(observation_trajectory)

 
    def _training_stage(self):
        """
        Parallel

        minibatch: set of observation trajectories
        """
        self.mu_zero.train_mode()
        for epoch in range(self.epochs):
            losses = []
            for batch_start in tqdm(range(0, self.minibatch_size, len(self.replay_buffer))):
                """
                k_step_policies: observed visist-counts used to compare with policy_prediction at each step k in the k-step rollout
                k_step_actions: selected actions from sample trajectory. Used for making hidden state transitions in the k-step rollout
                k_step_rewards: observed rewards used to compare with predicted rewards from the k-step rollout
                k_step_value: observed value from MCTS search used to compare with predicted value at each step k
                """
                self.mu_zero.optimizer.zero_grad()

                #prepare data
                with torch.no_grad():
                    batch_end = batch_start + self.minibatch_size
                    mbs_input_actions, mbs_input_states, k_step_policies, k_step_actions, k_step_rewards, k_step_value = self._prepare_minibatch(batch_start, batch_end, device)
                    input_actions_encoded = self._encode_actions(mbs_input_actions, self.state_history_length, self.real_resolution) #(batch, history_len, 16, 16)
                    k_step_actions_encoded = self._encode_actions(k_step_actions, self.K, self.latent_resolution) #these actions are used in the hidden state space (=latent_resolution)
                
                #forward pass
                predicted_reward, predicted_value, predicted_policy = self._k_step_rollout(mbs_input_states, input_actions_encoded, k_step_actions_encoded) #(batch, K, 601), ..., (batch, K, n_actions)
                bootstrapped_rewards = self._bootstrap_rewards(k_step_rewards, k_step_value)

                #backward
                loss = loss_fn(
                    observed_reward=k_step_rewards,
                    predicted_reward=predicted_reward,
                    bootstrapped_reward=bootstrapped_rewards,
                    predicted_value=predicted_value,
                    value_counts=k_step_policies, 
                    predicted_policy=predicted_policy,
                    target_transformation=self.mu_zero.rep_net._supports_representation, #TODO: improve
                    K=self.K
                )

                losses.append(loss.item())
                loss.backward()
                self.mu_zero.optimizer.step()

            #metrics + viz
            #mbs_input_states has shape (batch, 3*len_history, 16, 16)
            state_sequence = mbs_input_states[0].reshape(self.state_history_length, 3, self.real_resolution[0], self.real_resolution[1]) * 255
            # self.filewriter.add_image(f"trajectory_{epoch}", state_sequence, dataformats="NCHW")

            # Method 1: Add images with a step dimension for slider functionality
            for i in range(self.state_history_length):
                # Add each frame as a separate step in the same run
                self.filewriter.add_image(f"trajectory_{epoch}/frame", state_sequence[i], global_step=i, dataformats="CHW")
                
                # Also add text describing the action for this frame
                action = int(mbs_input_actions[0][i])
                self.filewriter.add_text(f"trajectory_{epoch}/action", f"Action: {action}", global_step=i)

            action_sequence = mbs_input_actions[0]
            action_log = "\n".join([f"Timestep {t}: Action {int(action_sequence[t])}" for t in range(self.state_history_length)])
            self.filewriter.add_text("trajectory_actions_" + str(epoch), action_log)

            print(f"Epoch {epoch} loss: {mean(losses)}, mean-reward: {mean(self.replay_buffer.get_reward_sums())}")

    def _prepare_minibatch(self, batch_start: int, batch_end: int, device: str): #TODO: no-grad???
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
        action_history_minibatch = self.replay_buffer.get_batched_past_actions(batch_start, batch_end)
        state_history_minibatch = self.replay_buffer.get_batched_states(batch_start, batch_end).view(self.minibatch_size, -1, self.real_resolution[0], self.real_resolution[1])
        
        #used for training loss
        k_step_policies = self.replay_buffer.get_batched_visit_counts(batch_start, batch_end)
        k_step_actions = self.replay_buffer.get_batched_future_actions(batch_start, batch_end) #NOTE: Tends to be mostly just 1s in the start (biased)
        k_step_rewards = self.replay_buffer.get_batched_rewards(batch_start, batch_end)
        k_step_values = self.replay_buffer.get_batched_values(batch_start, batch_end) 

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
            k_step_actions (batch_size, K, resolution[0], resolution[1]):  '

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
            hidden_state, reward = self.mu_zero.hidden_state_transition(hidden_state, k_step_actions[:, k, :, :].unsqueeze(1))
            reward_stack.append(reward)

        #save shit
        return torch.stack(reward_stack, dim=1), torch.stack(value_stack, dim=1), torch.stack(policy_stack, dim=1)

    def _bootstrap_rewards(self, rewards: torch.tensor, values: torch.tensor):
        """
        bootstrapped reward ≈ the value from the state
        n: number of steps into the future that is used to compute a bootstrapped target (distinct from k, but n = K for now)

        Args:
            rewards (batch_size, K):  
            values (batch_size, K):  

        Returns:
            bootstrapped_rewards (batch_size, K):  

            [
                sample_0: [bootstrap_0, bootstrap_1, bootstrap_2...],
                sample_1; [bootstrap_0, bootstrap_1, bootstrap_2...],
                ...
            ]
        """
        n = self.K
        discounts = self.gamma ** torch.arange(n, device=rewards.device, dtype=rewards.dtype).view(1, 1, -1) #(1, 1, n)

        rewards_repeat = rewards[:, None, :].repeat(1, n, 1) #create a new dimension with the repeated samples: (bs, K, K)
        mask_t = torch.triu(torch.ones((self.K, self.K))).to(device) #upper triangular matrix: (K, K)

        discounted_rewards = (rewards_repeat * mask_t) * discounts  
        discounted_rewards = discounted_rewards.sum(dim=2) #(bs, K)

        bootstrapped_values = values * (self.gamma ** n) #NOTE: This is wrong, this is  not v_t+n, but rather just v_t
        bootstrapped_rewards = discounted_rewards + bootstrapped_values
        return bootstrapped_rewards 

    def _sample_observation(self) -> torch.tensor:
        """
        Currently uses all sample observations in a Replay Buffer
        """ 
        return NotImplemented
    

    
from src.mcts import MCTSSearch
from src.networks import MuZeroAgent
from src.parallel_mcts import MCTSSearchVec
import time
from tqdm import tqdm
if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    #Args...
    planes = 256
    batch = 1 #currently gives the most speedup
    repnet_input = torch.ones((batch, planes, 4, 4)).to("cuda")
    
    #inits
    # muzero = MuZeroAgent(cfg=config["parameters"]["model"])
    
    # mcts_vec = MCTSSearchVec(cfg=config["parameters"], mu_zero=muzero)
    # print("STARTING VEC SEARCH")
    # start = time.time()
    # value, visit_counts = mcts_vec.search(repnet_input.to("cuda"), action_mask=torch.ones(batch, 3))
    # end = time.time()
    # print("FINISHED VEC SEARCH")
    # print(f"MCTS Vec time: {end - start}")

    # # print(value[:10])
    # # print(visit_counts[:10])
    
    # # search
    # repnet_input = torch.ones((1, planes, 4, 4))
    # mcts = MCTSSearch(cfg=config["parameters"], mu_zero=muzero)
    # print("STARTING SEARCH")
    # start = time.time()
    # for sample in tqdm(range(batch)):
    #     value, visit_counts = mcts.search(repnet_input.to("cuda"), action_mask=torch.tensor([0, 1, 1]).to(torch.float))
    # end = time.time()
    # print("FINISHED SEARCH")
    # print(f"MCTS seq time: {end - start}")

    trainer = RLSystem(config["parameters"])
    trainer.train()

    # mcts = TensorDictMCTSSearch(cfg=config["parameters"], mu_zero=muzero)
    # value, visit_counts = mcts.search(repnet_input, action_mask=torch.ones(batch, 3))


    print("Done")




"""
Todo:
- When we predict value/policy during MCTS search the outputs are softmax-distribution vectors 
- Fix tensor.to(device) abuse

Questions:
- Do we no_grad the episode generation?
- Do i need to move tensors to GPU for each iteration in the MCTS search -- device

Check:
- Normalization: Are the state inputs always in range [0, 1]
- BOTH hidden states and action tensors should be in range [0, 1]

Observations:
- When trajectory-length < 32 (often) we will always get a lot of zero-padded initialized history states as part of the repnet input
    ---> Solution: set self.state_history_length to smaller number 

"""
