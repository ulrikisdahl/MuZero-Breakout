import torch
from dataclasses import dataclass

@dataclass
class ObservationTrajectory:
    """
    Stores observation trajectories 
    """
    actions: list[int] 
    states: list[torch.tensor]
    rewards: list[float]
    visit_counts: list[torch.tensor]
    values: list[float] 
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


class ReplayBuffer:
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
                    value_target = observation_trajectory.get_values()[bootstrap_idx] * self.discount**self.K 
                    for k, reward in enumerate(observation_trajectory.get_rewards()[current_idx : bootstrap_idx]):
                        value_target += self.discount**k * reward
                else:
                    value_target = 0.0
                    for k, reward in enumerate(observation_trajectory.get_rewards()[current_idx : max_length]):
                        value_target += self.discount**k * reward
        
                bootstrapped_values_sample.append(value_target.item())
                bootstrap_idx += 1
            self.bootstrapped_values.append(torch.tensor(bootstrapped_values_sample)) 

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
        return torch.stack([self.past_actions_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_future_actions(self, batch_idxs: torch.tensor):
        """
        Returns actions used k-step rollout
        """
        return torch.stack([self.future_actions_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_states(self, batch_idxs: torch.tensor):
        """
        Returns a list of state-sequences[]

        Returns:
            tensor[]: tensor[batch_size, fixed_trajectory_length, channels, resolution[0], resolution[1]]
        """
        return torch.stack([self.state_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_rewards(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: tensor[batch_size, K] 
        """
        return torch.stack([self.reward_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_visit_counts(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: tensor[batch_size, K, n_actions]
        """
        return torch.stack([self.visit_counts_buffer[i] for i in batch_idxs.tolist()])

    def get_batched_values(self, batch_idxs: torch.tensor):
        """
        Returns:
            tensor[]: [batch_size, K, 1] 
        """
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
