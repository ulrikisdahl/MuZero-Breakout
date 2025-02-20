from src.networks import MuZeroAgent
import torch

#NOTE Does this take a lot more resources than just a dict?
class TreeStruct:
    def __init__(self, parent):
        self.parent = None
        self.children = {}
        self.value = None
        self.state = None


class MCTSSearch:
    def __init__(self, cfg, mu_zero: MuZeroAgent): 
        self.num_simulations = cfg["num_simulations"]
        self.actions = cfg["actions"]
        self.c1 = cfg["search"]["c1"]
        self.c2 = cfg["search"]["c2"]
        self.discount = cfg["search"]["discount"]
        self.mu_zero = mu_zero

    def search(self, hidden_state: torch.tensor, num_simulations: int, action_mask: torch.tensor): #TODO: use vmap here in JAX
        """
        Performs a latent MCTS search and returns the visit count statistics from the root node
        
        Args:
            actions (num_actions): list of possible actions to take
        
        tree: {
            state_0: {  
                {action-1}: { "N": 10, "Q": 0.5, "P": 0.2, "R": 1.0, "next_state": state_1 }, 
                {action-2}: { "N": 5, "Q": -0.2, "P": 0.3, "R": -1.0, "next_state": state_2 },
                ... ,
                "state": tensor,
                "value": int,
                "expanded": bool
            },
            state_0_1 {...}
        }
        """
        #Init root
        tree = {
            "state_0": {
                "state": hidden_state,
                "value": None,
                "expanded": False
            }
        }
        current_node = "state_0"
        prev_node = current_node
        subtree = tree[current_node] 
        action = self.actions[torch.randint(0, len(self.actions) + 1)]
        
        #prediction for root state
        policy_dist, value = self.mu_zero.evaluate_state(hidden_state)
        tree[current_node]["value"] = value

        #apply probability mask
        for action in self.actions:
            tree[current_node][action]["P"] = policy_dist[action] * action_mask[action] #action_mask[action] is zero if the action is not allowed
            tree[current_node][action]["next_state"] = f"state_{0}_{action}" #NOTE: this is a dummy state that will be expanded later

        #MCTS
        for simulation in range(self.num_simulations):
            trajectory = []
            current_node = "state_0"
            
            #traverse the tree
            while True:
                if not subtree["expanded"]: #time to expand   
                    subtree["expanded"] = True
                    current_node = subtree[action]["next_state"]
                    break

                #SELECT() 
                visit_sum = sum([subtree[act]["N"] for act in subtree])
                log_term = (visit_sum + self.c2 + 1) / self.c2
                ucb = torch.tensor([])
                for action in self.actions:
                    ucb_i = subtree[action]["Q"] + subtree[action]["P"] * torch.sqrt(visit_sum) / (1 + subtree[action]["N"]) * (self.c1 + torch.log(log_term))
                    torch.cat((ucb, torch.tensor([ucb_i])), dim=0)
                action = torch.argmax(ucb, dim=0) #select a_k
                
                prev_node = current_node
                current_node = subtree[action]["next_state"]
                subtree = tree[current_node]                     #NOTE: at the final step the node action points to does not yet have a real state
                trajectory.append((prev_node, action, subtree[prev_node][action]["R"]))
                    # --> Need to save prev_node because when we use the reward it is for the edge (s, a) going INTO current_node

            #EXPAND() - create state for the expand(node) and stats for children
            expanded_node_state, reward = self.mu_zero.hidden_state_transition(tree[prev_node]["state"], action)
            tree[current_node]["state"] = expanded_node_state
            tree[prev_node][action]["R"] = reward

            trajectory.append((prev_node, action, reward))

            policy_dist, value = self.mu_zero.evaluate_state(expanded_node_state) #wrong to evaluate on expanded state
            tree[current_node]["value"] = value 
            for action in self.actions:
                next_state = f"state_{simulation + 1}_{action}" 
                tree[current_node][action] = {"N": 0, "Q": 0, "P": policy_dist[action], "R": 0.0, "next_state": next_state}

            #BACKUP() - node-l = expanded node (current_node)
            for k, (node, action, r) in enumerate(trajectory):
                bootstrapped_reward = self._bootstrap_reward(trajectory, k, len(trajectory) + 1, value) 
                tree[node][action]["N"] += 1
                tree[node][action]["Q"] = (tree[node][action]["N"] * tree[node][action]["Q"] + bootstrapped_reward) / (tree[node][action]["N"] + 1)

        visit_counts = [tree["state_0"][action]["N"] for action in self.actions]
        
        #compute estimated value  ---  NOTE not so sure about this one
        value = 0.0 
        for action in self.actions:
            value += tree["state_0"][action]["N"] * tree["state_0"][action]["Q"]
        value /= sum(visit_counts)

        del hidden_tree, trajectory #NOTE: Dont need to keep the tree

        return value, visit_counts
        
    def _selection(self, sub_tree: dict):
        """
        UCB selection 
        """
        return
    
    def _expansion(self):
        """
        Create the hidden child nodes and statistics to their incoming edges (from expanded node) 
        """
        
        return
    
    def _backup(self):
        """
        Update the statistics along the simulation trajectory (tree path)
        """

        return

    def _bootstrap_reward(self, trajectory: list, k: int, depth: int, value_l: float):
        """
        Cumulative discounted reward
        """
        cumulative_discounted_reward = 0.0
        for tau in range(depth - 1 - k):
            cumulative_discounted_reward += torch.pow(self.discount, tau) * trajectory[k + 1 + tau][-1] + torch.pow(self.discount, depth - k) * value_l
        return cumulative_discounted_reward

    def k_step_rollout(self):
        """
        """
        #Maybe this does not belong to the MCTSSearch class

        return