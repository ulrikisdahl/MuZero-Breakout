import torch
import torch.nn as nn
import yaml
from utils import get_class, ReplayBuffer, ObservationTrajectory

reward_loss_fn = nn.NLLLoss() #used instead of CrossentropyLoss because the model outputs softmax probs
value_loss_fn = nn.NLLLoss #nn.MSELoss() NOTE: should be MSe
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
    """
    reward_loss = reward_loss(torch.log(predicted_reward), target_transformation(observed_reward)) #TODO: transform targets
    value_loss = value_loss_fn(torch.log(predicted_value), target_transformation(bootstrapped_reward)) #TODO: transform targets
    policy_loss = policy_loss_fn(predicted_policy, value_counts)
    return (1/K) * (reward_loss + value_loss + policy_loss)



class RLSystem:
    def __init__(self, cfg: dict):
        #hyperparams
        self.num_episodes = cfg["num_episodes"]
        self.num_steps = cfg["num_steps"] #??
        self.num_simulations = cfg["num_simulations"]
        self.minibatch_size = cfg["minibatch_size"]
        self.real_resolution = cfg["real_resolution"]
        self.latent_resolution = cfg["latent_resolution"]
        self.state_history_length = cfg["state_history_length"]
        self.actions = cfg["actions"]
        self.n_actions = len(self.actions)
        self.K = cfg["num_unroll_steps"]
        self.gamma = cfg["discount_factor"]

        #classes
        mu_zero_class = get_class("src.networks", cfg["model"]["agent_name"])
        self.mu_zero = mu_zero_class(cfg["model"]) 
        latent_mcts_class = get_class("src.mcts", cfg["search"]["mcts_name"])
        self.latent_mcts = latent_mcts_class(cfg["search"], self.mu_zero) #NOTE: passing the networks like this is not good
        environment_class = get_class(cfg["environment"]["environment_path"], cfg["environment"]["environment_name"])
        self.environment = environment_class(cfg["environment"]) 
            
        #other stuff
        self.replay_buffer = ReplayBuffer()
        self.observation_trajectory = ObservationTrajectory([], [], [], [], [], 0) #i = (state_i, action_i, reward_i, visit_counts_i, value_i)

    def train(self):
        """
        """
        epochs = 1
        for _ in range(epochs):
            #episode generation
            self._acting_stage()

            #network training
            self._training_stage()

    def _acting_stage(self):
        """
        Sequential
        """
        self.mu_zero.eval_mode()
        for episode in range(self.num_episodes):
            initial_state = self.environment.get_initial_state() / 255 #normalize
            self._reset_environment()

            self._run_episode(initial_state)

    def _run_episode(self, state: torch.tensor):
        """
        Runs an entire episode in the environment, from start to finish - Plays the game
        """
        done = False

        #At this level we are traversing the real environment
        while not done: #TODO: Add some max length
            value, visit_counts = self._sample_action(state) #π
            action = torch.argmax(visit_counts)
            state, reward, done = self.environment.step(state, action.item())
            state = state / 255 #normalize
            self.observation_trajectory.add_observation(
                action, state, reward, visit_counts, value
            )

        if self.observation_trajectory.length > (self.state_history_length + self.K): #only save trajectories with enough observations
            self.replay_buffer.append(self.observation_trajectory) #TODO: pad the array to get fixed shape? (must be tensor?)

    def _sample_action(self, state: torch.tensor):
        """
        Latent-MCTS + UCB

        Here we operate in the latent space, via the MCTS algorithm
        """
        repnet_input = self._prepare_mcts_input(state, self.real_resolution)
        hidden_state = self.mu_zero.hidden_state_transition(repnet_input)
        
        value, visit_counts = self.latent_mcts.search(hidden_state)
        return value, visit_counts

    def _prepare_mcts_input(self, state: torch.tensor, resolution):
        """
        Prepares a tensorized input from the sequence of states and actions

        Args:
            state (channels, resolution[0], resolution[1]): the current state

        Returns:
            repnet_input (1, 128, resolution[0], resolution[1]): encoded representation of the last 32 (32*3) states concatenated with last 32 actions 
        """
        actions = self.observation_trajectory.get_actions()[-self.state_history_length:].unsqueeze(0) 
        action_plane = self._encode_actions(torch.tensor(actions), resolution)

        state_sequence = self.observation_trajectory.get_states()[-self.state_history_length:].view(-1, resolution[0], resolution[1]) #(num_states*channels, res[0], res[1])
        state_sequence = torch.cat((state_sequence, state), dim=0).unsqueeze(0)

        repnet_input = torch.cat(state_sequence, action_plane, dim=1)
        return repnet_input
    
    def _encode_actions(self, actions: torch.tensor, resolution: tuple):
        """
        Takes in a list of actions and returns a tensor of shape (self.state_history_length, resolution[0], resolution[1])

        Args:
            actions (bs, actions)
            resolution (2): tuple of resolution
        """
        actions_expanded = (actions / self.n_actions)[:, :, None, None].expand(-1, -1, resolution[0], resolution[1]) #NOTE
        bias_plane = torch.ones((self.state_history_length, self.resolution[0], self.resolution[1])) *  actions_expanded
        return bias_plane

    def _reset_environment(self):
        """
        Sets the first 31 states in the trajectory to zero tensors
        
        self.state_history_length - 1: because the initial state is appended afterwards
        """
        self.observation_trajectory = [
            (torch.zeros(3, self.real_resolution[0], self.real_resolution[1]), 0, 0.0, 0, 0.0) #NOTE: action values always 0
            for _ in range(self.state_history_length - 1)
        ] 

 
    def _training_stage(self):
        """
        Parallel

        minibatch: set of observation trajectories
        """
        self.mu_zero.train_mode()

        for batch_start in range(0, self.minibatch_size, len(self.replay_buffer)):
            """
            k_step_policies: observed visist-counts used to compare with policy_prediction at each step k in the k-step rollout
            k_step_actions: selected actions from sample trajectory. Used for making hidden state transitions in the k-step rollout
            k_step_rewards: observed rewards used to compare with predicted rewards from the k-step rollout
            k_step_value: observed value from MCTS search used to compare with predicted value at each step k
            """
            self.mu_zero.optimizer.zero_grad()

            #prepare data
            batch_end = batch_start + self.minibatch_size
            mbs_input_actions, mbs_input_states, k_step_policies, k_step_actions, k_step_rewards, k_step_value = self._prepare_minibatch(batch_start, batch_end, device)
            
            #forward pass
            input_actions_encoded = self._encode_actions(mbs_input_actions, self.real_resolution)
            k_step_actions_encoded = self._encode_actions(k_step_actions, self.latent_resolution) #these actions are used in the hidden state space (=latent_resolution)
            predicted_reward, predicted_value, predicted_policy = self._k_step_rollout(mbs_input_states, input_actions_encoded, k_step_actions_encoded)
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
            loss.backward()
            self.mu_zero.optimizer.step()

    def _prepare_minibatch(self, batch_start: int, batch_end: int, device: str): #TODO: no-grad???
        """ 
        Needs to:
            Sample (uniformly) a current state and the previous 32 states + actions
            Sample next K rewards and value-counts (policy distributions)

        Returns:
            action_history_minibatch: action sequence up to current (root) state
            state_history_minibatch: state sequence up to current state 
            k_step_policies: K MCTS visit-counts statistics after current state 
            k_step_actions: K actions taken after current state
            k_step_rewards: K rewards gained after current state
            k_step_values: estimated values for K states after current
        """
        #used for RepNet input
        action_history_minibatch = self.replay_buffer.get_batched_past_actions(batch_start, batch_end)
        state_history_minibatch = self.replay_buffer.get_batch_states(batch_start, batch_start)
        
        #used for training loss
        k_step_policies = self.replay_buffer.get_batched_visit_counts(batch_start, batch_end)
        k_step_actions = self.replay_buffer.get_batched_future_actions(batch_start, batch_end) 
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
        repnet_input = torch.cat(input_states, input_actions, dim=1)
        hidden_state = self.mu_zero.create_hidden_state_root(repnet_input)

        policy_stack = []
        value_stack = []
        reward_stack = []
        for k in self.K:
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
            hidden_state, reward = self.mu_zero.hidden_state_transition(hidden_state, k_step_actions[:, k, :, :])
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
        mask_t = torch.triu(torch.ones((self.K, self.K))) #upper triangular matrix: (K, K)

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
if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    #Args...
    planes = 256
    repnet_input = torch.ones((1, planes, 6, 6))
    
    #inits
    muzero = MuZeroAgent(cfg=config["parameters"]["model"]) 
    mcts = MCTSSearch(cfg=config["parameters"], mu_zero=muzero)
    
    #search
    mcts.search(repnet_input, torch.tensor([1, 1, 0]).to(torch.float))
    

    print("Done")




"""
Todo:
- When we predict value/policy during MCTS search the outputs are softmax-distribution vectors 

Questions:
- Do we no_grad the episode generation?
- Do i need to move tensors to GPU for each iteration in the MCTS search -- device

Check:
- Normalization: Are the state inputs always in range [0, 1]
- BOTH hidden states and action tensors should be in range [0, 1]

"""
