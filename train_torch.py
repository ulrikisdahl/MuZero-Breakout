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
        self.resolution = cfg["resolution"]
        self.state_history_length = cfg["state_history_length"]
        self.actions = cfg["actions"]
        self.n_actions = len(self.actions)
        self.k = cfg["num_unroll_steps"]

        #classes
        mu_zero_class = get_class("src.networks", cfg["model"]["agent_name"])
        self.mu_zero = mu_zero_class(cfg["model"]) 
        latent_mcts_class = get_class("src.mcts", cfg["search"]["mcts_name"])
        self.latent_mcts = latent_mcts_class(cfg["search"], self.mu_zero) #NOTE: passing the networks like this is not good
        environment_class = get_class(cfg["environment"]["environment_path"], cfg["environment"]["environment_name"])
        self.environment = environment_class(cfg["environment"]) 
            
        #other stuff
        self.replay_buffer = ReplayBuffer()
        self.observation_trajectory = ObservationTrajectory() #i = (state_i, action_i, reward_i, visit_counts_i, value_i)

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

        if self.observation_trajectory.length > (self.state_history_length + self.k): #only save trajectories with enough observations
            self.replay_buffer.append(self.observation_trajectory) #TODO: pad the array to get fixed shape? (must be tensor?)

    def _sample_action(self, state: torch.tensor):
        """
        Latent-MCTS + UCB

        Here we operate in the latent space, via the MCTS algorithm
        """
        repnet_input = self._prepare_mcts_input(state)
        hidden_state = self.mu_zero.hidden_state_transition(repnet_input)
        
        value, visit_counts = self.latent_mcts.search(hidden_state)
        return value, visit_counts

    def _prepare_mcts_input(self, state: torch.tensor):
        """
        Prepares a tensorized input from the sequence of states and actions

        Args:
            state (channels, resolution[0], resolution[1]): the current state

        Returns:
            repnet_input (1, 128, resolution[0], resolution[1]): encoded representation of the last 32 (32*3) states concatenated with last 32 actions 
        """
        actions = self.observation_trajectory.get_actions()[-self.state_history_length:] #torch.Tensor([0, 1, 2])
        action_plane = self._encode_actions(torch.tensor(actions))

        state_sequence = self.observation_trajectory.get_states()[-self.state_history_length:].view(-1, self.resolution[0], self.resolution[1]) #(num_states*channels, res[0], res[1])
        state_sequence = torch.cat((state_sequence, state), dim=0) 

        repnet_input = torch.cat(state_sequence, action_plane, dim=0)
        return repnet_input.unsqueeze(0) #add batch dimension
    
    def _encode_actions(self, actions: torch.tensor):
        """
        Takes in a list of actions and returns a tensor of shape (self.state_history_length, resolution[0], resolution[1])
        """
        actions_expanded = (actions / self.n_actions)[:, None, None].expand(-1, self.resolution[0], self.resolution[1])
        bias_plane = torch.ones((self.state_history_length, self.resolution[0], self.resolution[1])) *  actions_expanded
        return bias_plane

    def _reset_environment(self):
        """
        Sets the first 31 states in the trajectory to zero tensors
        
        self.state_history_length - 1: because the initial state is appended afterwards
        """
        self.observation_trajectory = [
            (torch.zeros(3, self.resolution[0], self.resolution[1]), 0, 0.0, 0, 0.0) #NOTE: action values always 0
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
            predicted_reward, predicted_value, predicted_policy = self._k_step_rollout(mbs_input_states, mbs_input_actions, k_step_actions)

            #backward
            loss = loss_fn(
                observed_reward=k_step_rewards,
                predicted_reward=predicted_reward,
                bootstrapped_reward=k_step_value,
                predicted_value=predicted_value,
                value_counts=k_step_policies, 
                predicted_policy=predicted_policy,
                target_transformation=self.mu_zero.rep_net._supports_representation, #TODO: improve
                K=self.k
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

    def _k_step_rollout(self):
        """
        Recursively creates hidden states by calling Dynamics function K times, and evaluates each hidden state with Prediction function
        """
        return

    def _bootstrap_rewards(self):
        """
        bootstrapped reward ≈ the value from the state

        n: number of steps into the future that is used to compute a bootstrapped target (distinct from k)
        """
        return 

    def _sample_observation(self) -> torch.tensor:
        """
        Currently uses all sample observations in a Replay Buffer
        """ 
        return NotImplemented
    

    
if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    #Args...




"""
Todo:
- When we predict value/policy during MCTS search the outputs are softmax-distribution vectors 

Questions:
- Do we no_grad the episode generation?

Check:
- Normalization: Are the state inputs always in range [0, 1]
- BOTH hidden states and action tensors should be in range [0, 1]

"""
