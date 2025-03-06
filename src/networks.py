import torch
import torch.nn as nn
from utils import torch_activation_map

class ConvBlock(nn.Module):
    """
    """
    def __init__(self, activation: str, in_ch: int, out_ch: int, stride: int, kernel_size:int = 3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding) #NOTE: padding=1
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = torch_activation_map(activation)()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    Residual Block: H(x) = F(x) + x
    """
    def __init__(self, in_ch: int, activation: str):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,  kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,  kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.act = torch_activation_map(activation)()

    def forward(self, x):
        x1 = self.act(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x = x2 + x
        return self.act(x) 




class BaseNetwork(nn.Module):
    """
    Base class for the networks and wrapper for nn.Module class
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
        """
        Maps the reward/value to a more compact representation to compress large values into the support representation range
        Obtain categorical representations of the reward/value targets equivalent to the output representations of the networks
        """
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + self.epsilon * x)
    
    def _invertible_transform_compact_to_normal(self, x):
        """
        Maps the reward/value back to the original representation
        """
        return torch.sign(x) * ((torch.abs(x) + (1-self.epsilon))**2 - 1)

    def _supports_representation(self, target_value): #TODO: Move to MuZero class?
        """
        Rewards and Values represented categorically (one hot?) with a possible range of [-300, 300]
        Original value x is represented as x = p_low * x_low + p_high * x_high (page: 14)

        1. Transform target scalar using invertible transformation to compress
        2. Map it to the support set using a linear combination of two adjacent supports
        3. Return a probability distribution over the supports

        Args:
            target_value (batch_size, K): Observed rewards or values at each step k in the sample

        Return:
            support_vector (batch_size, K, 601): A probability distribution over the supports 
        """

        #transform to compact representation
        target_transformed = self._invertible_transform_normal_to_compact(target_value)
        
        #find the closes support indecies
        lower_idx = torch.searchsorted(self.supports, target_transformed, right=True) - 1
        lower_idx = lower_idx.clamp(0, self.num_supports - 1)
        upper_idx = (lower_idx + 1).clamp(0, self.num_supports - 1)

        #get the supports
        lower_support = self.supports[lower_idx]
        upper_support = self.supports[upper_idx]

        #compute linear combination coefficients
        p_low = (upper_support - target_transformed) / (upper_support - lower_support)
        p_high = 1 - p_low

        batch_size, k = target_value.shape
        support_vector = torch.zeros((batch_size, k, self.num_supports)).to(self.device)
        support_vector.scatter_(2, lower_idx.unsqueeze(-1), p_low.unsqueeze(-1))
        support_vector.scatter_(2, upper_idx.unsqueeze(-1), p_high.unsqueeze(-1))
        
        return support_vector #should be (bs, K, 601)
    
    def softmax_expectation(self, softmax_distribution):
        """
        Computes the expectation of a softmax distribution over the supports

        Used for inference
        """
        return torch.sum(softmax_distribution * self.supports, dim=1) #dim=1 for vectorized MCTS, dim=0 for sequential MCTS



class RepresentationNetwork(BaseNetwork):
    def __init__(self, cfg, in_ch: int):
        super().__init__(cfg)
        latent_ch = cfg["latent_channels"]
        activation = cfg["representation_network"]["activation"]
        num_res_blocks = cfg["representation_network"]["num_res_blocks"] #list specifying how many res-blocks should be in each sequence of res-blocks
        self.avg_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.blocks = nn.ModuleList([])
        self.blocks.append(
             nn.Conv2d(
                in_channels=in_ch,
                out_channels=latent_ch[0],
                kernel_size=(3,3),
                stride=1,
                padding=1
            )
        )
        for _ in range(num_res_blocks[0]): #first sequence of residual blocks
            self.blocks.append(
                ResidualBlock(
                    in_ch=latent_ch[0],
                    activation=activation
                )
            )
        
        self.blocks.append( #TODO: add activations after convs (even though paper doesnt specify it)
            nn.Conv2d(
                in_channels=latent_ch[0],
                out_channels=latent_ch[1],
                kernel_size=(3,3),
                stride=1,
                padding=1
            )
        )

        for _ in range(num_res_blocks[1]):
            self.blocks.append(
                ResidualBlock(
                    in_ch=latent_ch[1],
                    activation=activation                    
                )
            )
        
        self.blocks.append(self.avg_pool)
        
        for _ in range(num_res_blocks[2]):
            self.blocks.append(
                ResidualBlock(
                    in_ch=latent_ch[1],
                    activation=activation                    
                )
            )

        self.blocks.append(self.avg_pool)

    def forward(self, state):
        """
        """
        for module in self.blocks:
            state = module(state)
        return state



class DynamicsNetwork(BaseNetwork):
    """
    Input/Output: Hidden state + Action -> Hidden state + reward

    AlpaZero paper:
        - Each conv has 256 filters
    """
    def __init__(self, cfg: dict, in_ch: int, latent_resolution: int):
        super().__init__(cfg)
        self.num_res_blocks = cfg["dynamics_network"]["num_res_blocks"]
        activation = cfg["dynamics_network"]["activation"]
        num_supports = cfg["num_supports"] #601 supports in range [-300, 300] 
        # latent_resolution = (1,1) #TODO

        self.conv_block = ConvBlock(
            activation=activation,
            in_ch=in_ch + 1, # +1 for action encoding
            out_ch=in_ch,
            stride=1
        )

        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    in_ch=in_ch,
                    activation=activation                    
                )
            )

        #generates next hidden state
        self.state_head = nn.Sequential(
            #nothing - the state should be the same shape as input
        )

        #predicts the reward (expected softmax distribution over the supports for inference)
        self.reward_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch,
                out_ch=in_ch, #NOTE: should be 1 according to AlphaGo Appendix
                stride=1,
                kernel_size=1,
                padding=0
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=in_ch*latent_resolution[0]*latent_resolution[1], out_features=num_supports),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (batch, planes + action, resolution[0], resolution[1]): tensor of current hidden state concatenated with action encodings
        
        Returns:
            hidden_state (batch, planes, resolution[0], resolution[1]): generated hidden state
            reward (601,): probability distribution over supports representation
        """
        hidden_state = self.conv_block(hidden_state)
        for res_block in self.res_blocks:
            hidden_state = res_block(hidden_state)
        
        #predict reward
        reward = self.reward_head(hidden_state)

        return hidden_state, reward


class PredictionNetwork(BaseNetwork): 
    """
    Input/Output: Hidde state -> Policy dist, value est

    Args:
        cfg: network hyperparameters
        in_ch (int): size of input along channel dimension
        latent_res (int, int): resolution of input hidden state

    Returns:
        Policy: 
        Value: Estimate softmax supports distribution -> invert to scalar format using invertible transformation (page: 14)
    """
    def __init__(self, cfg: dict, in_ch: int, latent_resolution: tuple):
        super().__init__(cfg)
        self.num_res_blocks = cfg["prediction_network"]["num_res_blocks"]
        num_actions = cfg["prediction_network"]["num_actions"] #0: left, 1:stay, 2:right for breakout
        activation = cfg["prediction_network"]["activation"]
        num_supports = cfg["num_supports"]

        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    in_ch=in_ch,
                    activation=activation                    
                )
            )

        #generates a policy distribution
        self.policy_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch,
                out_ch=in_ch//2, #?
                stride=1
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=(in_ch//2) * latent_resolution[0] * latent_resolution[1], out_features=num_actions) #outputs a "distribution" over the possible actions
        ) #TODO: Remeber argmax!
 
        #generates a value estimate
        self.value_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch,
                out_ch=in_ch//2, 
                stride=1,
                kernel_size=1,
                padding=0
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            #NOTE: Accodring to AlphaGo Zero paper there should be another fully-connected layer here
            
            nn.Linear(in_features=(in_ch//2) * latent_resolution[0] * latent_resolution[1], out_features=num_supports),
            nn.Softmax(dim=1) #gives us the probability distribution over the support representation
        ) 

    def forward(self, hidden_state: torch.tensor):
        """
        Args:
            hidden_stae (batch, planes, resolution[0], resolution[1]): tensor of current hidden state without action planes
        
        Returns:
            policy_distribution (batch_size, num_actions): 
            value (batch_size, num_supports): 
        """
        for res_block in self.res_blocks:
            x = res_block(hidden_state)
        
        policy_distribution = self.policy_head(x)
        value = self.value_head(x)

        return policy_distribution, value

        
        
class MuZeroAgent(nn.Module):
    def __init__(self, cfg: dict):
        super(MuZeroAgent, self).__init__()
        real_state_planes = cfg["state_history_length"] * 3 + cfg["state_history_length"] #representation function input (page: 13)
        self.device = "cuda" #cfg["device"]
        self.rep_net = RepresentationNetwork(
            cfg=cfg,
            in_ch=real_state_planes
        )
        self.rep_net.to(self.device)
        self.dyn_net = DynamicsNetwork(
            cfg=cfg,
            in_ch=cfg["latent_channels"][1], 
            latent_resolution=cfg["latent_resolution"]
        )
        self.dyn_net.to(self.device)
        self.pred_net = PredictionNetwork(
            cfg=cfg,
            in_ch=cfg["latent_channels"][1], 
            latent_resolution=cfg["latent_resolution"] #TODO
        )
        self.pred_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg["learning_rate"])

    def create_hidden_state_root(self, state: torch.tensor):
        """
        Representation Network: Generates hidden root state from a real state

        Args:
            state (bs, state_history_length * 3 + state_history_length, resolution[0], resolution[1])
        """
        hidden_state_0 = self.rep_net(state.to(self.device)) #NOTE: device 
        return hidden_state_0
    
    def hidden_state_transition(self, prev_hidden_state: torch.tensor, action: torch.tensor):
        """
        Dynamics Network

        Action concatenated along channel (plane) dimension
        
        _encode_action here?
        """
        prev_hidden_state_with_action = torch.cat([prev_hidden_state, action], dim=1)
        hidden_state, reward = self.dyn_net(prev_hidden_state_with_action)
        return hidden_state, reward
    
    def evaluate_state(self, hidden_state: torch.tensor):
        """
        Prediction Network
        """
        policy_distribution, value = self.pred_net(hidden_state)
        return policy_distribution, value
    
    def _encode_action(self, action: int):
        """
        page 14: 'In Atari, an action is encoded as a one hot vector which is tiled appropriately into planes' 
        """
        return
    
    def eval_mode(self):
        """
        Set all networks to eval mode
        """
        self.rep_net.eval()
        self.dyn_net.eval()
        self.pred_net.eval()
    
    def train_mode(self):
        """
        Set all networks to train mode
        """
        self.rep_net.train()
        self.dyn_net.train()
        self.pred_net.train()


if __name__ == "__main__":
    """
    python3 -m src.networks
    """
    import yaml

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    muzero = MuZeroAgent(config["parameters"]["model"])

    hidden_state = torch.ones((32, 256, 6, 6))
    policy_distribution, value = muzero.evaluate_state(hidden_state)
    print(f"Policy distribution: {policy_distribution.shape}, Value: {value.shape}")

    normal_state = torch.ones(32, 128, 16, 16)
    hidden_state = muzero.create_hidden_state_root(normal_state)
    print(f"Hidden sate 0: {hidden_state.shape}")

    # prev_hidden_state = hidden_state
    # action = torch.ones((32, 1, 6, 6)) 
    # next_hidden_state = muzero.hidden_state_transition(prev_hidden_state, action)
    # print(f"Next hidden state: {next_hidden_state[0].shape}, Reward: {next_hidden_state[1].shape}")

    # print()
    # for param in muzero.parameters():
    #     print(param.shape)

    # print(f"N-params: {sum(1 for _ in muzero.parameters())}")
    # print("done")
    

"""
QUESTIONS

 - Supports are in range [-300, 300], but the invertible_transform(x) cant take in negative x values  --->  Forgot that formula uses torch.abs
    - Also paper says that prediction network outputs with Tanh-range (this is mainly said in AlphaGo Zero though)?
"""
