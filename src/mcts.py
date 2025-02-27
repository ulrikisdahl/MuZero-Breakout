from src.networks import MuZeroAgent
import torch
import math

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
        self.discount = cfg["search"]["discount_factor"]
        self.mu_zero = mu_zero
        self.latent_resolution = cfg["latent_resolution"]

    def search(self, hidden_state: torch.tensor, action_mask: torch.tensor): #TODO: use vmap here in JAX
        """
        Performs a latent MCTS search and returns the visit count statistics from the root node
        
        Args:
            hidden_state (batch_size=1, planes, resolution[0], resolution[1]): root node of the hidden state search tree
            action_mask (n_actions): 0 for illegal actions, 1 for legal actions
        
        tree: {
            state_0: {  
                0: { "N": 10, "Q": 0.5, "P": 0.2, "R": 1.0, "next_state": state_1 }, 
                1: { "N": 5, "Q": -0.2, "P": 0.3, "R": -1.0, "next_state": state_2 },
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
                0: {"N": 0, "Q": 0},
                1: {"N": 0, "Q": 0},
                2: {"N": 0, "Q": 0}, 
                "state": hidden_state,
                "value": None,
                "expanded": False
            }
        }
        current_node = "state_0"
        prev_node = current_node
        subtree = tree[current_node]
        batch_size = hidden_state.shape[0] 
        action = torch.multinomial(action_mask, num_samples=batch_size, replacement=True).item() #samples indecies of that have 1s in the mask
        
        #prediction for root state
        policy_dist, value = self.mu_zero.evaluate_state(hidden_state) #TODO: move to device ("cuda")
        policy_dist, value = torch.softmax(policy_dist.squeeze(0), dim=0), value.squeeze(0) #remove batch dim
        tree[current_node]["value"] = value

        #apply probability mask
        for action_i in self.actions:
            child_node = f"state_{0}_{action_i}"
            tree[current_node][action_i]["P"] = policy_dist[action_i] * action_mask[action_i] #action_mask[action] is zero if the action is not allowed
            tree[current_node][action_i]["next_state"] = child_node #NOTE: this is a dummy state that will be expanded later
            tree[child_node] = {"expanded": False}

        #s0 is now expaned (expect for the expanded field in its subtree dict -- soon...)

        #MCTS
        for simulation in range(self.num_simulations):
            trajectory = []
            current_node = "state_0"
            subtree = tree[current_node]
            
            #traverse the tree
            while True:
                if not subtree["expanded"]: #time to expand   
                    tree[current_node]["expanded"] = True
                    current_node = subtree[action]["next_state"]
                    tree[current_node] = { # expand the new node, but it is not fully expanded yet
                        0: {"N": 0},
                        1: {"N": 0},
                        2: {"N": 0}, 
                        "state": None,
                        "value": None,
                        "expanded": False
                    }
                    break

                #SELECT() 
                visit_sum = sum([subtree[act]["N"] for act in self.actions])
                log_term = (visit_sum + self.c2 + 1) / self.c2
                ucb = torch.tensor([])
                for action in self.actions:
                    ucb_i = subtree[action]["Q"] + subtree[action]["P"] * math.sqrt(visit_sum) / (1 + subtree[action]["N"]) * (self.c1 + math.log(log_term))
                    ucb = torch.cat((ucb, torch.tensor([ucb_i])), dim=0)
                action = torch.argmax(ucb, dim=0).item() #select a_k
                
                prev_node = current_node
                current_node = subtree[action]["next_state"]
                subtree = tree[current_node]                     #NOTE: at the final step the node action points to does not yet have a real state
                
                if subtree["expanded"]:
                    trajectory.append((prev_node, action, tree[prev_node][action]["R"]))
                    # --> Need to save prev_node because when we use the reward it is for the edge (s, a) going INTO current_node
                else:
                    tree[current_node] = { # expand the new node, but it is not fully expanded yet
                        0: {"N": 0},
                        1: {"N": 0},
                        2: {"N": 0}, 
                        "state": None,
                        "value": None,
                        "expanded": True #we will expand this node now
                    }
                    break

            #EXPAND() - create state for the expand(node) and stats for children    
            action_encoded = self._encode_action(action, self.latent_resolution, n_actions=3)
            expanded_node_state, reward = self.mu_zero.hidden_state_transition(tree[prev_node]["state"].to("cuda"), action_encoded.to("cuda"))
            reward = self.mu_zero.dyn_net.softmax_expectation(reward) #TODO: very unclean
            tree[current_node]["state"] = expanded_node_state
            tree[prev_node][action]["R"] = reward

            trajectory.append((prev_node, action, reward))

            policy_dist, value = self.mu_zero.evaluate_state(expanded_node_state.to("cuda")) #wrong to evaluate on expanded state
            policy_dist = torch.softmax(policy_dist.squeeze(0), dim=0)
            value = self.mu_zero.dyn_net.softmax_expectation(value) #TODO
            tree[current_node]["value"] = value 
            for action in self.actions:
                next_state = f"state_{simulation + 1}_{action}" 
                tree[current_node][action] = {"N": 0, "Q": 0, "P": policy_dist[action], "R": 0.0, "next_state": next_state}
                tree[next_state] = {"expanded": False}

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

        del tree, trajectory #NOTE: Dont need to keep the tree

        return value, torch.tensor(visit_counts)
        

    def _encode_action(self, action: int, resolution: tuple, n_actions: int): #TODO: Move to utils
        """
        Args:
            action (int): chosen action we want to encode

        Returns: 
            action_encoded (1, 1, res[0], res[1]): encoded version of the action scaled by 1/n_actions
        """
        action = torch.tensor([action])[None, None, None, :] #unsqueeze tensor
        action_encoded = action.expand(-1, -1, resolution[0], resolution[1]) * (1/n_actions)
        return action_encoded

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
        Cumulative discounted reward: Sums together all the discounted rewards that come after node-k, up until expanded node l 
        Adds the estimated value at node l to the sum 
        """
        cumulative_discounted_reward = 0.0
        for tau in range(depth - 1 - k):
            """
            We use reward from traj[k+tau] instead of traj[k+1+tau] because remember in SELECTION: R(s_l-1, a_l) = r_l  
            """
            cumulative_discounted_reward += (self.discount ** tau) * trajectory[k + tau][-1] + (self.discount ** (depth - k)) * value_l
        return cumulative_discounted_reward

    def k_step_rollout(self):
        """
        """
        #Maybe this does not belong to the MCTSSearch class

        return
    


"""
IMPROVEMENTS:
    - for scalar values we change between representing them as primitives and as torch tensors - consistency is needed

CONSIDERATIONS:
"""