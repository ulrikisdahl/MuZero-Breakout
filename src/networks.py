import torch
import torch.nn as nn
from utils import torch_activation_map
import torch.nn.functional as F
import torchvision.transforms as T

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


class RepresentationNetwork(nn.Module):
    def __init__(self, cfg, in_ch: int):
        super().__init__()
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
                # stride=2, #NOTE: Only for ALE breakout environment
                stride=1, 
                padding=1
            )
        )
        for _ in range(num_res_blocks[0]): 
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
                # stride=2, #NOTE: gym ALE environment
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



class DynamicsNetwork(nn.Module):
    """
    Input/Output: Hidden state + Action -> Hidden state + reward

    AlpaZero paper:
        - Each conv has 256 filters
    """
    def __init__(self, cfg: dict, in_ch: int, latent_resolution: int):
        super().__init__()
        self.num_res_blocks = cfg["dynamics_network"]["num_res_blocks"]
        activation = cfg["dynamics_network"]["activation"]
        num_supports = cfg["num_supports"] #601 supports in range [-300, 300] 
        num_actions = cfg["dynamics_network"]["num_actions"]

        self.conv_block = ConvBlock(
            activation=activation,
            in_ch=in_ch + num_actions, # +num_actions for action encoding
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
            nn.Linear(in_features=in_ch*latent_resolution[0]*latent_resolution[1], out_features=num_supports)
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


class PredictionNetwork(nn.Module):
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
        super().__init__()
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
        ) 
 
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
            nn.Linear(in_features=(in_ch//2) * latent_resolution[0] * latent_resolution[1], out_features=num_supports) #output raw logits
        ) 

    def forward(self, hidden_state: torch.tensor):
        """
        Args:
            hidden_stae (batch, planes, resolution[0], resolution[1]): tensor of current hidden state without action planes
        
        Returns:
            policy_distribution (batch_size, num_actions): 
            value (batch_size, num_supports): 
        """
        x = hidden_state
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy_distribution = self.policy_head(x)
        value = self.value_head(x)

        return policy_distribution, value


    
class MuZeroAgent(nn.Module):
    def __init__(self, cfg: dict):
        super(MuZeroAgent, self).__init__()
        real_state_planes = cfg["state_history_length"] * 1 + cfg["state_history_length"] #representation function input (page: 13)
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
            latent_resolution=cfg["latent_resolution"] 
        )
        self.pred_net.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg["learning_rate"], weight_decay=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda t: max(0.01 / 0.1, torch.exp(torch.tensor([-0.1 * t])).item())) #exponential decay


    def create_hidden_state_root(self, state: torch.tensor):
        """
        Representation Network: Generates hidden root state from a real state

        Args:
            state (bs, state_history_length * 3 + state_history_length, resolution[0], resolution[1])
        """
        # state = self.resize_transform(state) #NOTE: These two first lines are only necessary for ALE breakout environment
        # state = self._scale_state(state)
        hidden_state_0 = self.rep_net(state.to(self.device)) 
        hidden_state_0_scaled = self._scale_state(hidden_state_0)
        return hidden_state_0_scaled
    
    def hidden_state_transition(self, prev_hidden_state: torch.tensor, action: torch.tensor):
        """
        Dynamics Network

        Action concatenated along channel (plane) dimension

        Args:
            prev_hidden_state (batch_size, planes, latent_resolution[0], latent_resolution[1])

        Returns:
            hidden_state (batch_size, planes, latent_resolution[0], latent_resolution[1])
            reward (batch_size, )
        """
        prev_hidden_state_with_action = torch.cat([prev_hidden_state, action], dim=1)
        hidden_state, reward = self.dyn_net(prev_hidden_state_with_action)
        hidden_state_scaled = self._scale_state(hidden_state)
        return hidden_state_scaled, reward
    
    def evaluate_state(self, hidden_state: torch.tensor):
        """
        Prediction Network

        Args:
            hidden_state (batch_size, planes, latent_resolution[0], latent_resolution[1])

        Returns:
            policy_distribution (batch_size, n_actions)
            value (batch_size)
        """
        policy_distribution, value = self.pred_net(hidden_state)
        return policy_distribution, value
    
    def _scale_state(self, hidden_state: torch.tensor):
        """
        Returns:
            state_scaled (batch_size, planes, latent_resolution[0], latent_resolution[1])
        """
        hidden_state_flat = hidden_state.view(hidden_state.shape[0], -1) #flatten along all dimensions but batch-dim so that we can reduce across all the state dimensions
        s_min = hidden_state_flat.min(dim=1, keepdim=True)[0]
        s_max = hidden_state_flat.max(dim=1, keepdim=True)[0]
        
        #reshape to make the statistics broadcastable
        s_min = s_min.view(-1, 1, 1, 1)
        s_max = s_max.view(-1, 1, 1, 1)

        state_scaled = (hidden_state - s_min) / (s_max - s_min + 1e-8)
        return state_scaled
    
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
    import matplotlib.pyplot as plt
    import numpy as np

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    muzero = MuZeroAgent(config["parameters"]["model"])

    # hidden_state = torch.ones((32, 256, 4, 5)).to("cuda")
    # policy_distribution, value = muzero.evaluate_state(hidden_state)
    # print(f"Policy distribution: {policy_distribution.shape}, Value: {value.shape}")

    normal_state = torch.ones(32, 128, 16, 20).to("cuda")
    hidden_state = muzero.create_hidden_state_root(normal_state)
    print(f"Hidden sate 0: {hidden_state.shape}")

    plt.imshow(np.ones((3,3)))
    plt.show()

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
