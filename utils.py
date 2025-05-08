import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ObservationTrajectory:
    """
    Idea behind this class is to store each type of data more compactly such that retrieval is easier (avoids iteration)
    """
    actions: list[int] #TODO: init list?
    states: list[torch.tensor]
    rewards: list[float]
    visit_counts: list[torch.tensor]
    values: list[float] #NOTE: might be torch tensors
    length: int #only includes the real states of the trajectory (not zero paddings)
    reward_sum: int

    def add_observation(self, action, state, reward, visit_counts, values):
        """
        Adds a single observation tuple to the trajectory
        Increments the length counter

        Args:
            action (1, )
            state (3, 16, 16)
            reward (int)
            visit_counts (n_actions)
            values (1, )
        """
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.visit_counts.append(visit_counts)
        self.values.append(values) 
        self.reward_sum += reward
        self.length += 1

    def get_actions(self):
        """
        Returns:
            actions (n_history,)
        """
        return torch.tensor(self.actions) 
    
    def get_states(self):
        """
        Returns a tensor[] of tensors of shape (num_states, channels, resolution[0], resolution[1])
        """ 
        # return self.states
        return torch.stack(self.states) #(n_states, 3, 96, 96) - important to not merge the two first dims here

    def get_rewards(self):
        """
        Returns a tensor[] of floats
        """
        return torch.tensor(self.rewards)

    def get_visit_counts(self):
        """
        Returns a tensor of tensors: tensor[tensor[int]]
        """
        return torch.stack(self.visit_counts)

    def get_values(self):
        """
        Returns a tensor[] of floats
        """
        return torch.tensor(self.values)
    
    def get_reward_sum(self):
        """
        Returns tensor(1,)
        """
        return self.reward_sum


class ReplayBuffer: #TODO: make dataclass
    """
    Stores observation trajectory data, where each observation trajectory contains list[(action, state, reward, value_counts, value), ...]
    """
    def __init__(self, seq_len: int, K: int, max_length: int, discount: float, num_rewards_to_sum: int):   
        self.past_actions_buffer = [] #action-i leading to state-i
        self.future_actions_buffer = [] 
        self.state_buffer = [] #state-i
        self.reward_sums = []
        self.reward_buffer = [] #reward from (state_i-1, action_i)
        self.visit_counts_buffer = []
        self.value_buffer = [] #estimated value at state-i
        self.hist_seq_len = seq_len #includes the root state
        self.K = K
        self.length = 0
        self.max_length = max_length
        self.bootstrapped_values = []
        self.discount = discount
        self.num_rewards_to_sum = num_rewards_to_sum

    def save_observation_trajectory(self, observation_trajectory: ObservationTrajectory): 
        """
        Each time a state and action trajectory is saved to their respective buffers, only a fixed length chunk of those trajectories are saved. This allows us to tensorize

        state_start: the root state for the MCTS simulation (the uniform state sampling happens here)

        Args:
            observation: (action, state, reward, visit_counts, value)
        """
        # if observation_trajectory.length > 40: #no need to sample from earlier states then (NOTE: can lead to catastrophic forgetting when the agent becomes very good)
        #     state_start = torch.randint(self.hist_seq_len*2, self.hist_seq_len*2 + (observation_trajectory.length - self.hist_seq_len) - self.K, (1,)).item()
        # else:
        #     state_start = torch.randint(self.hist_seq_len, self.hist_seq_len + observation_trajectory.length - self.K + 1, (1,)).item() #NOTE: +1 makes it so we can actually get the negative reward 

        # trajectory_start = state_start - self.hist_seq_len  #TODO: Wrong   <----- Current code can train on initial padded history only

        #create a sample of every state in the observation trajectory
        for state in range(observation_trajectory.length - self.K + 1):
            state_start = state + self.hist_seq_len
            trajectory_start = state_start - self.hist_seq_len
            
            self.past_actions_buffer.append(
                #select a fixed length of actions from the trajectory such that this returned list can be tensorized
                observation_trajectory.get_actions()[trajectory_start : state_start] 
            )
            self.future_actions_buffer.append(
                observation_trajectory.get_actions()[state_start : state_start + self.K]
            )
            self.state_buffer.append(
                #each observation trajectory is a fixed length
                observation_trajectory.get_states()[trajectory_start : state_start]
            )

            #save rewards for metrics
            self.reward_sums.append(observation_trajectory.get_reward_sum().item()) #no need to index because we want all the collected rewards

            #used for k-step rollout
            self.reward_buffer.append(
                observation_trajectory.get_rewards()[state_start : state_start + self.K]
            )
            self.visit_counts_buffer.append(
                observation_trajectory.get_visit_counts()[state_start : state_start + self.K]
            )
            self.value_buffer.append(
                observation_trajectory.get_values()[state_start : state_start + self.K]
            )
            
            #create value targets
            td_steps = 10
            max_length = self.hist_seq_len + observation_trajectory.length
            bootstrap_idx = state_start + td_steps
            bootstrapped_values_sample = []
            for current_idx in range(state_start, state_start + self.K):
                if (bootstrap_idx < max_length):
                    value_target = observation_trajectory.get_values()[bootstrap_idx] * self.discount**self.K       #NOTE It is correct to use [bootstrap_idx] and not [bootstrap_idx-1] because the mcts-value is saved along with its successor state
                    for k, reward in enumerate(observation_trajectory.get_rewards()[current_idx : bootstrap_idx]):
                        value_target += self.discount**k * reward
                else:
                    value_target = 0.0
                    for k, reward in enumerate(observation_trajectory.get_rewards()[current_idx : max_length]):
                        value_target += self.discount**k * reward
        
                bootstrapped_values_sample.append(value_target.item())
                bootstrap_idx += 1
            self.bootstrapped_values.append(torch.tensor(bootstrapped_values_sample)) #NOTE: Forgot to pop these!

            self.length += 1
            if self.length > self.max_length:
                self.past_actions_buffer.pop(0)
                self.future_actions_buffer.pop(0)
                self.state_buffer.pop(0)
                self.reward_buffer.pop(0)
                self.visit_counts_buffer.pop(0)
                self.value_buffer.pop(0)
                self.reward_sums.pop(0)
                self.bootstrapped_values.pop(0)
                self.length -= 1
                
    def get_batched_past_actions(self, batch_idxs: torch.tensor):
        """
        Returns actions used as input to RepNet

        Returns:
            tensor[]: tensor[batch_size, fixed_trajectory_length]
        """
        # return torch.stack(self.past_actions_buffer[batch_start:batch_end])
        return torch.stack([self.past_actions_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_future_actions(self, batch_idxs: torch.tensor):
        """
        Returns actions used k-step rollout
        """
        # return torch.stack(self.future_actions_buffer[batch_start:batch_end])
        return torch.stack([self.future_actions_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_states(self, batch_idxs: torch.tensor):
        """
        Returns a list of state-sequences[]

        Returns:
            tensor[]: tensor[batch_size, fixed_trajectory_length, channels, resolution[0], resolution[1]]
        """
        # return torch.stack(self.state_buffer[batch_start:batch_end]) #(batch_size, fixed, channels, resolution[1], resolution[2])
        return torch.stack([self.state_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_rewards(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: tensor[batch_size, K] 
        """
        # return torch.stack(self.reward_buffer[batch_start:batch_end])
        return torch.stack([self.reward_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_visit_counts(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: tensor[batch_size, K, n_actions]
        """
        # return torch.stack(self.visit_counts_buffer[batch_start:batch_end]) 
        return torch.stack([self.visit_counts_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_values(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: [batch_size, K, 1] 
        """
        # return torch.stack(self.value_buffer[batch_start:batch_end])
        # return torch.stack([self.value_buffer[i] for i in batch_idxs.tolist()])
        return torch.stack([self.bootstrapped_values[i] for i in batch_idxs.tolist()])

    def get_reward_sums(self):
        """
        Gives the reward sums for the last 512 samples (newest episode generation stage)
        """
        return self.reward_sums[-self.num_rewards_to_sum:]

    def empty_buffer(self):
        """
        """
        self.past_actions_buffer = [] #action-i leading to state-i
        self.future_actions_buffer = [] 
        self.state_buffer = [] #state-i
        self.reward_sums = []
        self.reward_buffer = [] #reward from (state_i-1, action_i)
        self.visit_counts_buffer = []
        self.value_buffer = [] #estimated value at state-i
        self.length = 0
        self.bootstrapped = []

    def __len__(self):
        return self.length



class ScalarTransforms(nn.Module):
    """
    Class for representing scalar values as support distributions
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.epsilon = 0.001
        supports_min = cfg["supports_min"]
        supports_max = cfg["supports_max"]
        self.num_supports = cfg["num_supports"]
        self.device = cfg["device"]
        self.supports = torch.linspace(supports_min, supports_max, self.num_supports).to(self.device)

    def _invertible_transform_normal_to_compact(self, x):
        """ Maps the reward/value to a more compact representation to compress large values into the support representation range
        Obtain categorical representations of the reward/value targets equivalent to the output representations of the networks """
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + self.epsilon * x)
        
    def _invertible_transform_compact_to_normal(self, x):
        """ Maps the reward/value back to the original representation """
        return torch.sign(x) * ((torch.abs(x) + (1-self.epsilon))**2 - 1)
        
    def supports_representation(self, target_value):
        """ Rewards and Values represented categorically with a possible range of [-300, 300]
        Original value x is represented as x = p_low * x_low + p_high * x_high (page: 14)
        1. Transform target scalar using invertible transformation to compress
        2. Map it to the support set using a linear combination of two adjacent supports
        3. Return a probability distribution over the supports
        
        Args:
            target_value (batch_size, K): Observed rewards or values at each step k in the sample
            
        Return:
            support_vector (batch_size, K, num_supports): A probability distribution over the supports
        """
        # Transform to compact representation
        target_transformed = self._invertible_transform_normal_to_compact(target_value)
        
        # Find the closest support indices
        lower_idx = torch.searchsorted(self.supports, target_transformed, right=True) - 1
        lower_idx = lower_idx.clamp(0, self.num_supports - 2)  # Fix 3: Ensure upper_idx doesn't go out of bounds
        upper_idx = lower_idx + 1
        
        # Get the supports
        lower_support = self.supports[lower_idx]
        upper_support = self.supports[upper_idx]
        
        # Compute linear combination coefficients
        p_low = (upper_support - target_transformed) / (upper_support - lower_support + 1e-10)
        p_high = 1 - p_low
        
        batch_size, k = target_value.shape
        support_vector = torch.zeros((batch_size, k, self.num_supports)).to(self.device)
        support_vector.scatter_(2, lower_idx.unsqueeze(-1), p_low.unsqueeze(-1))
        support_vector.scatter_(2, upper_idx.unsqueeze(-1), p_high.unsqueeze(-1))
        
        return support_vector
    
    def _softmax_expectation(self, softmax_distribution):
        """
        Computes the expectation of a softmax distribution over the supports

        Used for inference
        """
        return torch.sum(softmax_distribution * self.supports, dim=-1)

    def inverted_softmax_expectation(self, softmax_distribution):
        """
        First computes the expected value under the respective "softmax" distribution and subsequently inverts the scaling transformation
        """
        softmax_distribution = F.softmax(softmax_distribution, dim=-1)
        softmax_expectation = self._softmax_expectation(softmax_distribution)
        inverted_transform = self._invertible_transform_compact_to_normal(softmax_expectation)
        return inverted_transform


def get_class(module_name: str, class_name: str):
    """
    Dynamically imports and retreives classes

    Args:
        module_name (path): module location
        class_name (string): name of the class
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except:
        raise ImportError(f"Could not import module {module_name}")
    

def torch_activation_map(activation: str) -> nn.Module:
    """
    Returns the torch activation function based on the string input
    """
    return {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU
    }[activation]