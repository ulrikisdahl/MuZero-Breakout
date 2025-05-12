"""
Tensorized version of breakout that can be parallelized on the GPU
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class MuZeroEnvironment(ABC):
    """Abstract base class for MuZero environments."""
    
    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Returns the initial state of the environment."""
        pass
    
    @abstractmethod
    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Executes one step in the environment.
        
        Args:
            state: Current state tensor
            action: Action to take
            
        Returns:
            tuple of (next_state, reward, done)
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns a binary tensor indicating valid actions for the given state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Binary tensor of shape (num_actions,) where 1 indicates valid action
        """
        pass
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Returns the size of the action space."""
        pass
    
    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the state tensor."""
        pass


class BreakoutEnvironment(MuZeroEnvironment):
    """
    Tensorized implementation of Breakout for MuZero
    
    States:
        ball_x (int): X-coordinate of the ball in a traditional plane
        ball_y (int): Y-coordinate of the ball in a traditional plane
    """
    def __init__(
        self,
        cfg: dict, 
        width: int = 10,
        height: int = 15,
        paddle_width: int = 6,
        brick_rows: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.height = 16 #cfg["resolution"][0]
        self.width = 20 #cfg["resolution"][1]
        self.paddle_width = paddle_width
        self.brick_rows = 3 #cfg["brick_rows"]
        self.device = "cpu" 
        self.batch = cfg["n_parallel"] #the amount of games we suimulate in parallel
        self.paddle_hit_reward = cfg["paddle_hit_reward"]
        self.brick_hit_reward = cfg["brick_hit_reward"]
        self.game_lost_reward = cfg["game_lost_reward"]
        self.game_won_reward = cfg["game_won_reward"]
        
        # Define constants for state tensor channels
        self.CHANNEL_PADDLE = 0
        self.CHANNEL_BALL = 1
        self.CHANNEL_BRICKS = 2
        
        # Actions: 0 = left, 1 = stay, 2 = right
        self._action_space_size = 3
        
        # Ball direction is maintained in environment state
        self.ball_dx = 1
        self.ball_dy = -1
        
    @property
    def action_space_size(self) -> int:
        return self._action_space_size
    
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.batch, 3, self.height, self.width)
    
    def reset(self) -> torch.Tensor:
        """
        state (batch, 3, heigh, width)
        """
        state = torch.zeros(self.state_shape, device=self.device)
        
        #randomize the starting position
        random_low = -6 #pos starts at middel 
        random_high = self.width - self.paddle_width - (self.width // 2 - self.paddle_width // 2 - 1)
        random_init_offset = torch.randint(random_low, random_high, (self.batch,), device=self.device)
        # random_init_offset = torch.zeros((self.batch), device=self.device).int() #uncomment for fixed simulations

        # Place paddle at bottom center
        paddle_pos = self.width // 2 - self.paddle_width // 2 + random_init_offset
        
        batch_idxs = torch.arange(self.batch, device=self.device).unsqueeze(1)
        state[batch_idxs, self.CHANNEL_PADDLE, -1, paddle_pos.unsqueeze(1) + torch.arange(self.paddle_width, device=self.device).unsqueeze(0)] = 1  #for last dim we index by explicitly giving all the indecies we want to use instead of splicing
        
        # Place ball just above paddle (or dont!)
        random_ball_pos = torch.randint(1, self.width - 1, (self.batch,), device=self.device)
        random_ball_height = torch.randint(-3, -1, (self.batch,), device=self.device)
        state[batch_idxs, self.CHANNEL_BALL, random_ball_height.unsqueeze(1) + torch.arange(1, device=self.device).unsqueeze(0), random_ball_pos.unsqueeze(1) + torch.arange(1, device=self.device).unsqueeze(0)] = 1
        
        # Place bricks at top
        state[:, self.CHANNEL_BRICKS, :self.brick_rows, :] = 1
        
        #set dx to either 1 or -1 randomly
        dx_init_values = torch.tensor([-1, 1], device=self.device)
        # dx_init_values = torch.tensor([1, 1], device=self.device) #uncomment for fixed simulations
        self.ball_dx = torch.tensor([dx_init_values[torch.randint(0, 2, (1,)).item()] for _ in range(self.batch)], device=self.device)
        self.ball_dy = torch.ones(self.batch, device=self.device) * -1
        
        return state, 0 #0 is a dummy
    
    def get_valid_actions(self, state: torch.Tensor, paddle_pos_new: torch.Tensor) -> torch.Tensor:
        """
        valid [(move_left, stay, move_right), ...]

        Args:
            state (batch, 3, height, width)
            paddle_pos_new (batch, )

        Returns:
            valid (batch, n_actions)
        """
        valid = torch.ones((self.batch, self.action_space_size), device=self.device)
        valid[(paddle_pos_new == 0), 0] = 0
        valid[(paddle_pos_new + self.paddle_width >= self.width), -1] = 0        
        return valid
    
    # @torch.compile
    def step(self, state: torch.Tensor, action: torch.Tensor, done_mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one step in the environment for every sample, vectorized.
        
        Args:
            state (batch, 3, height, width): current state before we transition
            action (batch,): one action per state, corresponding to the action you take from that state
            done_mask (batch,): mask indicating which samples are already finished (and should just be padded with zeros)

        Returns:
            next_state (batch, 3, height, width)
            reward (batch, )
            done_mask (batch, )
            valid_actions (batch, n_actions)
        """
        next_state = state.clone()
        reward = torch.zeros((self.batch, ), device=self.device)

        #move paddle    
        paddle_pos = torch.argmax(state[:, self.CHANNEL_PADDLE, -1, :], dim=1) #argmax picks the first it finds
        paddle_pos_new = paddle_pos + torch.where(action == 0, -1, torch.where(action == 2, 1, 0)) #where action==0: -1, or else do (torch.where(action==2) where action==2 do +1) 
        paddle_pos_new = paddle_pos_new.clamp(min=0, max=self.width - self.paddle_width)
        
        batch_idx = torch.arange(next_state.shape[0], device=next_state.device).unsqueeze(1)
    
        next_state[:, self.CHANNEL_PADDLE, -1, :] = 0 #remove paddle
        paddle_offsets = torch.arange(self.paddle_width, device=next_state.device).unsqueeze(0)
        paddle_positions = paddle_pos_new.unsqueeze(1) + paddle_offsets #(batch, paddle_width)
        next_state[batch_idx, self.CHANNEL_PADDLE, -1, paddle_positions] = 1

        #get ball position
        ball_pos = torch.where(state[:, self.CHANNEL_BALL] == 1) 
        batch_idx, ball_y, ball_x = ball_pos[0], ball_pos[1], ball_pos[2] #3 x torch.Tensor([16]) 
        ball_x, ball_y = ball_x.to(torch.float), ball_y.to(torch.float)


        #precompute which balls need to flip dx before moving
        wall_hit_mask = torch.logical_or(ball_x + self.ball_dx < 0, ball_x + self.ball_dx >= self.width)
        self.ball_dx = torch.where(wall_hit_mask, -self.ball_dx, self.ball_dx)
        # move ball using current direction 
        new_ball_y = ball_y + self.ball_dy
        new_ball_x = ball_x + self.ball_dx
        
        #check if game is lost
        missed_ball_mask = new_ball_y >= self.height  # Ball has fallen past the paddle row
        reward[missed_ball_mask] = self.game_lost_reward  # Penalize missing the ball
        done_mask |= missed_ball_mask 
        next_state[done_mask, self.CHANNEL_BRICKS] = 0   #reset states for finished games
        next_state[done_mask, self.CHANNEL_PADDLE] = 0
        self.ball_dx[done_mask] = 0
        self.ball_dy[done_mask] = 0
        new_ball_y[missed_ball_mask] = 0
        
        
        #ceiling collision
        self.ball_dy[new_ball_y < 0] *= -1
        new_ball_y[new_ball_y < 0] = ball_y[new_ball_y < 0] #resets the y position to the previous position (but form now on ball_dy points in the opposite direction)

        #brick collision
        old_ball_dy = self.ball_dy.clone()
        new_ball_x_bricks = new_ball_x - (new_ball_x % 2) #group pixels along x-axis to create concept of brick
        brick_mask = next_state[batch_idx, self.CHANNEL_BRICKS, new_ball_y.int(), new_ball_x_bricks.int()] == 1
        self.ball_dy = torch.where(brick_mask, -old_ball_dy, self.ball_dy) #reverse velocity
        next_state[batch_idx, self.CHANNEL_BRICKS, new_ball_y.int(), new_ball_x_bricks.int()] = 0 #remove collided bricks
        next_state[batch_idx, self.CHANNEL_BRICKS, new_ball_y.int(), (new_ball_x_bricks + 1).int()] = 0
        # Reflect the ball's vertical position using the original velocity
        new_ball_y = torch.where(brick_mask, ball_y - old_ball_dy, new_ball_y)
        # Add reward for hitting a brick
        reward[brick_mask.bool()] += self.brick_hit_reward

        #paddle collision
        paddle_row_hit = new_ball_y == (self.height - 1)
        
        paddle_mask = torch.zeros_like(next_state[:, self.CHANNEL_PADDLE, -1, :])
        paddle_mask.scatter_(1, paddle_positions, 1) #gives us the paddle mask for each sample

        ball_hits_paddle = paddle_row_hit & paddle_mask[batch_idx, new_ball_x.int()].int() #if ball is both at last row and in the area of the paddle mask --> paddle hit
        self.ball_dy = torch.where(ball_hits_paddle.to(torch.bool), -self.ball_dy, self.ball_dy)

        paddle_center = paddle_pos_new + self.paddle_width // 2
        hit_offset = new_ball_x - paddle_center
        reward[ball_hits_paddle.bool()] += self.paddle_hit_reward                                   #TODO: could lead to reward hacking 

        #clear previous ball position and assign new ball position
        next_state[:, self.CHANNEL_BALL] = 0  
        next_state[batch_idx, self.CHANNEL_BALL, new_ball_y.int(), new_ball_x.int()] = 1

        #check terminal state
        game_finished = ~next_state[:, self.CHANNEL_BRICKS].any(dim=(1, 2)) #(batch, )
        done_mask |= game_finished #1 if already set or game just finished (game_finished)
        next_state[done_mask, self.CHANNEL_BRICKS] = 0 #need to have some value in CHANNEL_BALL in order to not collapse ball_x dimension
        next_state[done_mask, self.CHANNEL_PADDLE] = 0
        reward[game_finished ^ missed_ball_mask] += self.game_won_reward #XOR: make sure that we only add reward if the game was finished by the ball NOT falling out of screen

        valid_actions = self.get_valid_actions(next_state, paddle_pos_new)

        return next_state, reward, done_mask, valid_actions
    
    
    def render(self, state: torch.Tensor) -> str:
        """Returns a string representation of two states side-by-side for debugging."""
        assert state.shape[0] == 2, "Input state must have shape (2, 3, height, width)"

        canvas = []
        for y in range(self.height):
            row_1 = "¦"
            row_2 = "¦"

            for x in range(self.width):
                # First state
                if state[0, self.CHANNEL_BRICKS, y, x] == 1:
                    row_1 += "█"
                elif state[0, self.CHANNEL_BALL, y, x] == 1:
                    row_1 += "●"
                elif state[0, self.CHANNEL_PADDLE, y, x] == 1:
                    row_1 += "="
                else:
                    row_1 += " "

                # Second state
                if state[1, self.CHANNEL_BRICKS, y, x] == 1:
                    row_2 += "█"
                elif state[1, self.CHANNEL_BALL, y, x] == 1:
                    row_2 += "●"
                elif state[1, self.CHANNEL_PADDLE, y, x] == 1:
                    row_2 += "="
                else:
                    row_2 += " "

            row_1 += "¦"
            row_2 += "¦"

            # Combine both rows with space separator
            canvas.append(row_1 + "   " + row_2)

        return "\n".join(canvas)
