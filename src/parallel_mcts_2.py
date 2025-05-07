from src.networks import MuZeroAgent
import torch
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

class MCTSSearchVec:

    def __init__(self, cfg, mu_zero: MuZeroAgent):
        self.num_simulations = cfg["num_simulations"]
        self.actions = cfg["actions"]
        self.c1 = cfg["search"]["c1"]
        self.c2 = cfg["search"]["c2"]
        self.discount = cfg["search"]["discount_factor"]
        self.mu_zero = mu_zero
        self.latent_resolution = cfg["latent_resolution"]
        self.dirchlet_alpha = 0.25
        self.noise_weight = 0.175
        self.decay_steps = 30 #number of steps until dirichlet noise is zero

    def search(self, hidden_state: torch.tensor, action_mask: torch.tensor, training_iteration: int):
        """
        Performs a latent MCTS search and returns the visit count statistics from the root node
        Parallel: Traverses each sample's MCTS tree sequentially each simulation-iteration, but builds up a "buffer"
                  of expanded states in order to call the networks in parallel on GPU

        Args:
            hidden_state (parallel, planes, resolution[0], resolution[1]): root node of the hidden state search tree
            action_mask (parallel, n_actions): 0 for illegal actions, 1 for legal actions
            training_iteration: which iteration we are on
        
        Returns:
            value (batch_size): The estimated value from the root node
            visit_counts (batch_size, 3): How many times each action was chosen from the root node
        """
        batch_size = hidden_state.shape[0]
        
        # Initialize trees and root nodes
        trees = self._initialize_trees(batch_size, hidden_state)
        
        # Expand root nodes
        trees, expand_buffer, last_nodes, actions, trajectories = self._expand_root_nodes(
            trees, hidden_state, action_mask, batch_size
        )
        
        # Perform MCTS simulations
        for simulation in range(self.num_simulations):
            if simulation > 0:
                # Select nodes to expand
                trajectories, expand_buffer, last_nodes = self._select_nodes(
                    trees, batch_size, action_mask
                )
            
            # Expand selected nodes and get updated states
            expanded_node_states, rewards, policy_dists, values = self._expand_nodes(
                expand_buffer, last_nodes
            )
            
            # Update statistics and backup values
            self._backup(
                trees, last_nodes, trajectories, expanded_node_states, 
                rewards, values, policy_dists, simulation, batch_size
            )
        
        # Compute final results
        values, visit_counts_batched = self._compute_results(trees, batch_size)
        
        return torch.tensor(values), torch.stack(visit_counts_batched)
    
    def _initialize_trees(self, batch_size, hidden_state):
        """Initialize the MCTS trees for each batch sample"""
        trees = [
            {
                "state_0": {
                    0: {"N": 0, "Q": 0},
                    1: {"N": 0, "Q": 0},
                    2: {"N": 0, "Q": 0},
                    3: {"N": 0, "Q": 0},
                    "state": hidden_state[sample],
                    "value": None,
                    "expanded": False
                }
            }
            for sample in range(batch_size)
        ]
        return trees
    
    def _expand_root_nodes(self, trees, hidden_state, action_mask, batch_size):
        """Predict and expand the root nodes of the search trees"""
        # Prediction for root state
        with torch.no_grad():
            policy_dist, value = self.mu_zero.evaluate_state(hidden_state)
            
        value = self.mu_zero.rep_net.inverted_softmax_expectation(value).to("cpu")
        policy_dist = policy_dist.to("cpu")
        policy_dist_valid = policy_dist.clone()
        # policy_dist_valid[action_mask==0] = float("-inf") #NOTE: Remove for now...
        policy_dist_valid = torch.softmax(policy_dist_valid, dim=1)
        
        # Apply probability mask and select initial action
        actions = []
        trajectories = []
        expand_buffer = []
        last_nodes = []
        
        for idx, tree in enumerate(trees):
            current_node = "state_0"
            tree[current_node]["value"] = value[idx]
            tree[current_node]["expanded"] = True
            
            # Dirichlet noise
            noise = torch.distributions.Dirichlet(torch.ones(len(self.actions)) * self.dirchlet_alpha).sample()
            
            # Set up statistics for children
            for action_i in self.actions:
                child_node = f"state_{0}_{action_i}"
                tree[current_node][action_i]["P"] = (1-self.noise_weight) * policy_dist_valid[idx][action_i] + self.noise_weight * noise[action_i]
                tree[current_node][action_i]["next_state"] = child_node
                tree[child_node] = {"expanded": False}
                
            trees[idx] = tree
            action_i = self.ucb_action(tree[current_node], torch.ones_like(action_mask), idx)
            actions.append(action_i)
            
            # Use initial action
            prev_node = current_node
            current_node = tree[current_node][action_i]["next_state"]
            expand_buffer.append(hidden_state[idx])
            last_nodes.append((prev_node, action_i, current_node))
            trajectories.append([])
            
        return trees, expand_buffer, last_nodes, actions, trajectories
    
    def _select_nodes(self, trees, batch_size, action_mask):
        """
        Traverses the tree from each sample using UCB for selection until un-expanded node is reached

        Returns:
            trajectories: 
            expand_buffer: 
            last_nodes:
        """
        trajectories = []
        expand_buffer = []
        last_nodes = []
        
        for sample in range(batch_size):
            trajectory = []
            current_node = "state_0"
            prev_node = current_node
            subtree = trees[sample][current_node]
            
            level = 0
            while True:
                action_i = self.ucb_action(subtree, torch.ones_like(action_mask), sample)
                
                prev_node = current_node
                current_node = subtree[action_i]["next_state"]
                subtree = trees[sample][current_node]
                
                level += 1
                if subtree["expanded"]:
                    trajectory.append((prev_node, action_i, trees[sample][prev_node][action_i]["R"]))
                else:
                    trees[sample][current_node] = {
                        0: {"N": 0},
                        1: {"N": 0},
                        2: {"N": 0},
                        3: {"N": 0},
                        "state": None,
                        "value": None,
                        "expanded": True  # We will expand this node now
                    }
                    expand_buffer.append(trees[sample][prev_node]["state"])
                    last_nodes.append((prev_node, action_i, current_node))
                    break
                    
            trajectories.append(trajectory)
            
        return trajectories, expand_buffer, last_nodes
    
    def _expand_nodes(self, expand_buffer, last_nodes):
        """
        Computes state tensor and transition reward for all expanded nodes in parallel
        Computes estimated value and policy distribution for all expanded nodes in parallel
        """
        actions_precoded = torch.tensor([action for _, action, _ in last_nodes])
        action_encoded = self._encode_action(actions_precoded, self.latent_resolution, n_actions=len(self.actions)).to("cuda")
        expand_buffer = torch.stack(expand_buffer)
        
        with torch.no_grad():
            expanded_node_states, rewards = self.mu_zero.hidden_state_transition(expand_buffer, action_encoded)
            policy_dists, values = self.mu_zero.evaluate_state(expanded_node_states)
            
        rewards = self.mu_zero.rep_net.inverted_softmax_expectation(rewards)
        values = self.mu_zero.rep_net.inverted_softmax_expectation(values)
        policy_dists = torch.softmax(policy_dists.to("cpu"), dim=1)
        
        return expanded_node_states, rewards, policy_dists, values
    
    def _backup(self, trees, last_nodes, trajectories, expanded_node_states, rewards, values, policy_dists, simulation, batch_size):
        """
        Update node statistics and backup values through the tree
        """
        for sample in range(batch_size):
            prev_node = last_nodes[sample][0]
            action = last_nodes[sample][1]
            current_node = last_nodes[sample][2]
            value = values[sample]
            
            # Update expanded node
            trees[sample][current_node]["state"] = expanded_node_states[sample]
            trees[sample][prev_node][action]["R"] = rewards[sample]
            trees[sample][current_node]["value"] = value
            
            # Create edges for potential next states from expanded node
            for action_i in self.actions:
                next_state = f"state_{simulation + 1}_{action_i}"
                trees[sample][current_node][action_i] = {
                    "N": 0, "Q": 0, "P": policy_dists[sample][action_i], 
                    "R": 0.0, "next_state": next_state
                }
                trees[sample][next_state] = {"expanded": False}
                
            # Add transition stats from prev_node to current_node
            trajectories[sample].append((prev_node, action, rewards[sample]))
            
            # Backup values through the trajectory
            for k, (node, action_i, r) in reversed(list(enumerate(trajectories[sample]))):
                value = value * self.discount + r
                trees[sample][node]["value"] += value.to("cpu")
                trees[sample][node][action_i]["Q"] = (trees[sample][node][action_i]["N"] * trees[sample][node][action_i]["Q"] + value) / (trees[sample][node][action_i]["N"] + 1)
                trees[sample][node][action_i]["N"] += 1
    
    def _compute_results(self, trees, batch_size):
        """
        Compute the estimated value for the root node along with visit counts statistics
        """
        values = []
        visit_counts_batched = []
        
        for sample in range(batch_size):
            visit_counts = [trees[sample]["state_0"][action]["N"] for action in self.actions]
            visit_counts_batched.append(torch.tensor(visit_counts))
            values.append(
                trees[sample]["state_0"]["value"].item() / self.num_simulations
            )
            
        return values, visit_counts_batched
    
    def _encode_action(self, action: torch.Tensor, resolution: tuple, n_actions: int):
        """
        Encodes a batch of actions into a tensor with n_actions planes, as per MuZero. Same as used _encode_action_dynamics

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
        action_encoded = action_one_hot.view(batch_size, n_actions, 1, 1).expand(-1, -1, resolution[0], resolution[1]) #tile across spatial dimension
        return action_encoded

    def _bootstrap_reward(self, trajectory: list, k: int, depth: int, value_l: float):
        """
        Cumulative discounted reward: Sums together all the discounted rewards that come after node-k, up until expanded node l 
        Adds the estimated value at node l to the sum 
        """
        cumulative_discounted_reward = 0.0
        for tau in range(depth - 1 - k):
            cumulative_discounted_reward += (self.discount ** tau) * trajectory[k + tau][-1] 
        cumulative_discounted_reward += (self.discount ** (depth - k - 1)) * value_l
        return cumulative_discounted_reward

    def ucb_action(self, subtree, action_mask, idx):
        """
        Selects the action given the ucb formula
        """
        visit_sum = sum([subtree[act]["N"] for act in self.actions])
        log_term = (visit_sum + self.c2 + 1) / self.c2
        ucb = []
        for action_i in self.actions:
            ucb_i = subtree[action_i]["Q"] + subtree[action_i]["P"] * math.sqrt(visit_sum) / (1 + subtree[action_i]["N"]) * (self.c1 + math.log(log_term))
            if not action_mask[idx][action_i]: #action is illegal from root state
                ucb_i = -math.inf
            ucb.append(ucb_i)
        ucb = torch.tensor(ucb)
        max_val = ucb.max().item()
        
        best_indices = (ucb == max_val).nonzero(as_tuple=True)[0]
        selected_index = best_indices[torch.randint(len(best_indices), (1,))].item() #introduce some randomness if there are ties
        return selected_index