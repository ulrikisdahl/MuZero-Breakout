import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class MuZeroEnvironment(ABC):
    """Abstract base class for MuZero environments."""
    
    @abstractmethod
    def get_initial_state(self) -> torch.Tensor:
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


class MuZeroEnvironment(ABC):
    """Abstract base class for MuZero environments."""
    
    @abstractmethod
    def get_initial_state(self) -> torch.Tensor:
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
        width: int = 10,
        height: int = 15,
        paddle_width: int = 4,
        brick_rows: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.brick_rows = brick_rows
        self.device = device
        
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
        return (3, self.height, self.width)
    
    def get_initial_state(self) -> torch.Tensor:
        state = torch.zeros(self.state_shape, device=self.device)
        
        #randomize the starting position
        random_init_offset = torch.randint(-2, 3, (1,)).item()

        # Place paddle at bottom center
        paddle_pos = self.width // 2 - self.paddle_width // 2 + random_init_offset
        state[self.CHANNEL_PADDLE, -1, paddle_pos:paddle_pos + self.paddle_width] = 1
        
        # Place ball just above paddle
        state[self.CHANNEL_BALL, -2, self.width // 2 + random_init_offset] = 1
        
        # Place bricks at top
        state[self.CHANNEL_BRICKS, :self.brick_rows] = 1
        
        # Reset ball direction to initial state (up-right)

        #set dx to either 1 or -1 randomly
        dx_init_values = torch.tensor([-1, 1])
        dx_idx = torch.randint(0, 2, (1,))
        self.ball_dx = dx_init_values[dx_idx].item()
        self.ball_dy = -1
        
        return state
    
    def get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        paddle_pos = torch.where(state[self.CHANNEL_PADDLE].any(dim=0))[0][0].item()
        
        valid = torch.ones(self.action_space_size, device=self.device)
        if paddle_pos == 0:  # Left edge
            valid[0] = 0  # Can't move left
        if paddle_pos + self.paddle_width >= self.width:  # Right edge
            valid[2] = 0  # Can't move right
            
        return valid
    
    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, float, bool]:
        next_state = state.clone()
        reward = 0
        done = False
        
        # 1. Move paddle
        paddle_pos = torch.where(state[self.CHANNEL_PADDLE].any(dim=0))[0][0].item()
        if action == 0 and paddle_pos > 0:  # Move left
            next_state[self.CHANNEL_PADDLE] = torch.roll(state[self.CHANNEL_PADDLE], -1, dims=1)
        elif action == 2 and paddle_pos + self.paddle_width < self.width:  # Move right
            next_state[self.CHANNEL_PADDLE] = torch.roll(state[self.CHANNEL_PADDLE], 1, dims=1)
        
        # 2. Get ball position
        ball_pos = torch.where(state[self.CHANNEL_BALL] == 1)
        if len(ball_pos[0]) == 0:  # Ball lost
            return next_state, -1.0, True
            
        ball_y, ball_x = ball_pos[0][0], ball_pos[1][0]
        
        # 3. Move ball using current direction
        new_ball_y = ball_y + self.ball_dy
        new_ball_x = ball_x + self.ball_dx
        print(f"Y: {new_ball_y}")
        print(f"X: {new_ball_x}")
        
        # 4. Handle collisions
        # Wall collisions
        if new_ball_x < 0 or new_ball_x >= self.width:
            self.ball_dx *= -1
            new_ball_x = ball_x + self.ball_dx
            
        if new_ball_y < 0:
            self.ball_dy *= -1
            new_ball_y = ball_y + self.ball_dy
        
        # Paddle collision
        if new_ball_y == self.height - 1:
            paddle_range = next_state[self.CHANNEL_PADDLE, -1].nonzero().squeeze()
            if new_ball_x in paddle_range:
                self.ball_dy *= -1
                
                # Modify angle based on where ball hits paddle
                paddle_center = paddle_pos + self.paddle_width // 2
                hit_offset = new_ball_x - paddle_center

                # Change horizontal direction based on where ball hits paddle
                self.ball_dx = 1 if hit_offset >= 0 else -1 #NOTE: has to be "hit_offset >= 0" if we have even number pixels for padle
                # new_ball_y = ball_y + self.ball_dy #NOTE: delay the reaction such that it doesnt turn around until next step
            else:
                return next_state, -1.0, True  # Ball lost
        
        # Brick collision
        if next_state[self.CHANNEL_BRICKS, new_ball_y, new_ball_x] == 1:
            #group a brick into pixels of two
            new_ball_x_bricks = new_ball_x - (new_ball_x % 2) #always the left-most pixel of the two-pixel brick
            next_state[self.CHANNEL_BRICKS, new_ball_y, new_ball_x_bricks:new_ball_x_bricks + 2] = 0
            self.ball_dy *= -1
            new_ball_y = ball_y + self.ball_dy #NOTE: stops working if we remove this
            reward = 1.0
        
        # Update ball position
        next_state[self.CHANNEL_BALL] = torch.zeros_like(state[self.CHANNEL_BALL])
        next_state[self.CHANNEL_BALL, new_ball_y, new_ball_x] = 1
        
        # Check if all bricks are destroyed
        if not next_state[self.CHANNEL_BRICKS].any():
            return next_state, reward + 5.0, True
        
        return next_state, reward, done
    
    def render(self, state: torch.Tensor) -> str:
        """Returns a string representation of the state for debugging."""
        canvas = []
        for y in range(self.height):
            row = "¦"
            for x in range(self.width):
                if state[self.CHANNEL_BRICKS, y, x] == 1:
                    row += "█"
                elif state[self.CHANNEL_BALL, y, x] == 1:
                    row += "●"
                elif state[self.CHANNEL_PADDLE, y, x] == 1:
                    row += "="
                else:
                    row += " "
            row += "¦"
            canvas.append(row)
        return "\n".join(canvas)



if __name__ == "__main__":
    breakout = BreakoutEnvironment()
    next_state = breakout.get_initial_state()
    print(next_state[1, :, :])

    while True:
        action = input()
        if action == "a":
            next_state, reward, done = breakout.step(next_state, 0)
        elif action == "d":
            next_state, reward, done = breakout.step(next_state, 2)
        else:
            next_state, reward, done = breakout.step(next_state, 1)

        print(breakout.render(next_state))
        # plt.imshow(next_state.permute(1, 2, 0).cpu().numpy())
        # plt.show()

    # for _ in range(25):

    #     if _ == 2 or _ == 3 or _ == 4:
    #         next_state, reward, done = breakout.step(next_state, 2)
    #     else:
    #         next_state, reward, done = breakout.step(next_state, 1)
    #     print(done)
    


    #     plt.imshow(next_state.permute(1, 2, 0).cpu().numpy())
    #     plt.show()