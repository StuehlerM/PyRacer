"""
Neural network models for reinforcement learning.
"""
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Safe config import with fallback defaults
try:
    from utils.config import config
    DEFAULT_INPUT_DIM = config.STATE_DIM
    DEFAULT_OUTPUT_DIM = config.ACTION_DIM
    DEFAULT_HIDDEN_DIM = config.HIDDEN_DIM
except (ImportError, AttributeError):
    DEFAULT_INPUT_DIM = 8
    DEFAULT_OUTPUT_DIM = 4
    DEFAULT_HIDDEN_DIM = 64


class DQN(nn.Module):
    """
    Deep Q-Network for the racing game.
    
    A simple feed-forward neural network that outputs Q-values for each action.
    """
    
    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, 
                 output_dim: int = DEFAULT_OUTPUT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        """
        Initialize the DQN.
        
        Args:
            input_dim: int - dimension of input state
            output_dim: int - number of actions (output Q-values)
            hidden_dim: int - number of neurons in hidden layers
            
        Raises:
            ValueError: If any dimension is not positive
        """
        super(DQN, self).__init__()
        
        # Input validation
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions (input_dim, output_dim, hidden_dim) must be positive integers")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: torch.Tensor - input state tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor - Q-values for each action, shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: numpy.ndarray - current state array
            epsilon: float - exploration probability (0 to 1)
        
        Returns:
            int: selected action index
        """
        if random.random() < epsilon:
            # Random action (exploration)
            return random.randint(0, self.output_dim - 1)
        else:
            # Greedy action (exploitation)
            with torch.inference_mode():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    
    Separates value and advantage streams to improve learning.
    This can help the agent learn which states are valuable without
    having to learn the effect of each action for each state.
    """
    
    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, 
                 output_dim: int = DEFAULT_OUTPUT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        """
        Initialize the Dueling DQN.
        
        Args:
            input_dim: int - dimension of input state
            output_dim: int - number of actions
            hidden_dim: int - number of neurons in hidden layers
            
        Raises:
            ValueError: If any dimension is not positive
        """
        super(DuelingDQN, self).__init__()
        
        # Input validation
        if input_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions (input_dim, output_dim, hidden_dim) must be positive integers")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: torch.Tensor - input state tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor - Q-values for each action, shape (batch_size, output_dim)
        """
        features = self.feature_layer(x)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s))
        # Detach mean to prevent gradient flow through it (common practice)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True).detach())
        
        return q_values
    
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: numpy.ndarray - current state array
            epsilon: float - exploration probability (0 to 1)
        
        Returns:
            int: selected action index
        """
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.inference_mode():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()


class ConvDQN(nn.Module):
    """
    Convolutional DQN for image-based states.
    
    This would be used if we represent the state as an image (e.g., from camera).
    For our current implementation, we use vector states, so this is optional.
    Uses adaptive pooling to handle varying input image sizes.
    """
    
    def __init__(self, input_channels: int = 3, 
                 output_dim: int = DEFAULT_OUTPUT_DIM):
        """
        Initialize the ConvDQN.
        
        Args:
            input_channels: int - number of input channels (e.g., 3 for RGB)
            output_dim: int - number of actions
            
        Raises:
            ValueError: If input_channels or output_dim are not positive
        """
        super(ConvDQN, self).__init__()
        
        if input_channels <= 0 or output_dim <= 0:
            raise ValueError("input_channels and output_dim must be positive integers")
        
        self.output_dim = output_dim
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Adaptive pooling ensures consistent output size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: torch.Tensor - input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor - Q-values for each action, shape (batch_size, output_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling ensures consistent feature size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy for image-based state.
        
        Args:
            state: numpy.ndarray - current state image array of shape (channels, height, width)
            epsilon: float - exploration probability (0 to 1)
        
        Returns:
            int: selected action index
        """
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.inference_mode():
                # Convert numpy array to tensor and add batch dimension
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()


def test_model():
    """Test all models to ensure they work correctly."""
    print("Testing models...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test DQN
    print("\n=== Testing DQN ===")
    model = DQN(input_dim=10, output_dim=5, hidden_dim=64)
    assert model.device == device, f"Model device mismatch: {model.device} vs {device}"
    
    # Create a random input
    x = torch.randn(32, 10).to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output type: {type(output)}")
    print(f"Output device: {output.device}")
    
    assert output.shape == (32, 5), f"Expected output shape (32, 5), got {output.shape}"
    assert output.device == device, f"Output device mismatch: {output.device} vs {device}"
    
    # Test act() method
    state = np.random.randn(10).astype(np.float32)
    action_greedy = model.act(state, epsilon=0.0)  # Always greedy
    assert 0 <= action_greedy < 5, f"Action {action_greedy} out of range [0, 4]"
    
    action_random = model.act(state, epsilon=1.0)  # Always random
    assert 0 <= action_random < 5, f"Action {action_random} out of range [0, 4]"
    
    print("DQN model test passed!")
    
    # Test Dueling DQN
    print("\n=== Testing DuelingDQN ===")
    model = DuelingDQN(input_dim=10, output_dim=5, hidden_dim=64)
    assert model.device == device, f"Model device mismatch: {model.device} vs {device}"
    
    output = model(x)
    print(f"Dueling DQN output shape: {output.shape}")
    assert output.shape == (32, 5), f"Expected output shape (32, 5), got {output.shape}"
    assert output.device == device, f"Output device mismatch: {output.device} vs {device}"
    
    # Test act() method
    action = model.act(state, epsilon=0.0)
    assert 0 <= action < 5, f"Action {action} out of range [0, 4]"
    
    print("Dueling DQN test passed!")
    
    # Test ConvDQN
    print("\n=== Testing ConvDQN ===")
    model = ConvDQN(input_channels=3, output_dim=5)
    assert model.device == device, f"Model device mismatch: {model.device} vs {device}"
    
    # Test with different image sizes (thanks to adaptive pooling)
    test_sizes = [(32, 3, 84, 84), (16, 3, 100, 100), (8, 3, 64, 64)]
    for batch_size, channels, height, width in test_sizes:
        x_img = torch.randn(batch_size, channels, height, width).to(device)
        output = model(x_img)
        print(f"  Input: {x_img.shape} -> Output: {output.shape}")
        assert output.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {output.shape}"
        assert output.device == device, f"Output device mismatch: {output.device} vs {device}"
    
    # Test ConvDQN act() method
    state_img = np.random.randn(3, 84, 84).astype(np.float32)
    action = model.act(state_img, epsilon=0.0)
    assert 0 <= action < 5, f"Action {action} out of range [0, 4]"
    
    print("ConvDQN test passed!")
    
    # Test input validation
    print("\n=== Testing Input Validation ===")
    try:
        DQN(input_dim=0, output_dim=5, hidden_dim=64)
        assert False, "Should have raised ValueError for input_dim=0"
    except ValueError as e:
        print(f"  Caught expected error for invalid input_dim: {e}")
    
    try:
        DQN(input_dim=10, output_dim=-1, hidden_dim=64)
        assert False, "Should have raised ValueError for output_dim=-1"
    except ValueError as e:
        print(f"  Caught expected error for invalid output_dim: {e}")
    
    print("Input validation test passed!")
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_model()
