import torch
import torch.nn as nn
from utils import torch_activation_map

class ConvBlock(nn.Module):
    """
    """
    def __init__(self, activation: str, in_ch: int, out_ch: int, stride: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), stride=stride, padding=1) #NOTE: padding=1
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
    Base class for the networks
    """
    def __init__(self, ):
        super().__init__()
        self.epsilon = 0.001
        self.supports = torch.tensor([i for i in range(-300, 301)], dtype=torch.float32)

    def _invertible_transform(self, x):
        """
        Obtain categorical representations of the reward/value targets equivalent to the output representations of the networks
        """
        return torch.sgn(x) * (torch.sqrt(x + 1) - 1 + self.epsilon * x)
    
    def _supports_transform(self, x):
        """
        A support is a 
        Rewards and Values represented categorically (one hot?) with a possible range of [-300, 300]

        3.7 = 0.3*3 + 0.7*4 
        """

class RepresentationNetwork(BaseNetwork):
    def __init__(self, cfg, in_ch: int):
        super().__init__()


class DynamicsNetwork(BaseNetwork):
    """
    Input/Output: Hidden state + Action -> Hidden state + reward
    """
    def __init__(self, cfg: dict, in_ch: int):
        super().__init__()
        self.num_res_blocks = cfg["num_res_blocks"]
        res_block_act = cfg["res_block_act"]
        activation = cfg["activation"]
        supports_range = cfg["supports_range"] #601 supports in range [-300, 300] 
        latent_resolution = (1,1) #TODO

        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    in_ch=in_ch,
                    activation=res_block_act                    
                )
            )

        #generates next hidden state
        self.state_head = nn.Sequential(
            #nothing - the state should be the same as input
        )

        #predicts the reward (expected softmax distribution for the supports)
        self.reward_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch,
                out_ch=in_ch
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=in_ch*latent_resolution[0]*latent_resolution[1], out_features=supports_range),
            nn.Softmax(dim=1)
        )

    def forwad(self, x):
        """
        Args:
            x (batch, planes, resolution[0], resolution[1]): tensor of current hidden state concatenated with action encodings
        """
        return x


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
    def __init__(self, cfg: dict, in_ch: int, latent_res: tuple):
        super().__init__()
        self.num_res_blocks = cfg["num_res_blocks"]
        res_block_act = cfg["res_block_act"]
        num_actions = cfg["num_actions"] #0: left, 1:stay, 2:right for breakout
        activation = cfg["activation"]
        supports_range = cfg["supports_range"]

        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    in_ch=in_ch,
                    activation=res_block_act                    
                )
            )

        #generates a policy distribution
        self.policy_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch,
                out_ch=in_ch//2
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=(in_ch//2) * latent_res[0] * latent_res[1], out_features=num_actions) #outputs a probability distribution over the possible actions
        ) #TODO: Remeber argmax!
 
        #generates a value estimate
        self.value_head = nn.Sequential(
            ConvBlock(
                activation=activation,
                in_ch=in_ch//2
            ),
            nn.Flatten(start_dim=1, end_dim=-1),
            #NOTE: Accodring to AlphaGo Zero paper there should be another fully-connected layer here
            # nn.Linear(in_features=(in_ch//2) * latent_res[0] * latent_res[1], out_features=1), #outputs estimated return of reward from current state
            
            nn.Lienar(in_features=(in_ch//2) * latent_res[0] * latent_res[1], out_features=supports_range),

            # nn.Tanh() #[-1=lose, 1=win], (AlphaGo Zero) 
        ) 

    def forward(self, x: torch.tensor):
        """
        Args:
            x (batch, planes, resolution[0], resolution[1]): tensor of current hidden state without action planes
        """
        

class MuZeroAgent(nn.Module):
    def __init__(self, cfg: dict):
        super(MuZeroAgent, self).__init__()
        self.rep_net = RepresentationNetwork()
        self.dyn_net = DynamicsNetwork()
        self.pred_net = PredictionNetwork()

    def state_transition(self, prev_hidden_state: torch.tensor, action: torch.tensor):
        """
        """
        return
    
    def evaluate_state(self, hidden_state: torch.tensor):
        """
        """
        return
    
    def create_hidden_space(self, state: torch.tensor):
        """
        """
        return

    def forward(self, x):
        return
    
    def _encode_action(self, action: int):
        """
        page 14: 'In Atari, an action is encoded as a one hot vector which is tiled appropriately into planes' 
        """
        return
    


"""
QUESTIONS

 - Supports are in range [-300, 300], but the invertible_transform(x) cant take in negative x values? 
    - Also paper says that prediction network outputs with Tanh-range (this is mainly said in AlphaGo Zero)?
"""