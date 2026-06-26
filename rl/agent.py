"""
DQN Agent for reinforcement learning.
Implements the Deep Q-Network algorithm with experience replay.
"""
import logging
from typing import Optional, Dict, Any, Union

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .model import DQN, DuelingDQN
from .memory import ReplayBuffer
from utils.config import config


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network.
    
    Implements the DQN algorithm as described in:
    "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
    
    Includes:
    - Epsilon-greedy exploration
    - Experience replay buffer
    - Target network for stable training
    - Double DQN improvement option
    - Dueling DQN architecture option
    - Polyak averaging for target network updates
    """
    
    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        action_dim: int = config.ACTION_DIM,
        lr: float = config.LEARNING_RATE,
        gamma: float = config.GAMMA,
        epsilon: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        epsilon_decay: float = config.EPSILON_DECAY,
        memory_size: int = config.MEMORY_SIZE,
        batch_size: int = config.BATCH_SIZE,
        target_update_freq: int = config.TARGET_UPDATE_FREQ,
        use_dueling: bool = False,
        use_double_dqn: bool = True,
        polyak_tau: float = 0.005,
        grad_clip_norm: float = 1.0,
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: int - dimension of state
            action_dim: int - number of actions
            lr: float - learning rate
            gamma: float - discount factor (must be in (0, 1])
            epsilon: float - initial exploration rate
            epsilon_min: float - minimum exploration rate
            epsilon_decay: float - exploration decay rate per step
            memory_size: int - size of replay buffer
            batch_size: int - batch size for training
            target_update_freq: int - steps between target network updates
            use_dueling: bool - use Dueling DQN architecture
            use_double_dqn: bool - use Double DQN for target calculation
            polyak_tau: float - interpolation factor for Polyak averaging (0 = hard update)
            grad_clip_norm: float - maximum norm for gradient clipping
            
        Raises:
            AssertionError: If any parameter is invalid
        """
        # Validate inputs
        assert state_dim > 0, "state_dim must be positive"
        assert action_dim > 0, "action_dim must be positive"
        assert 0 < gamma <= 1, "gamma must be in (0, 1]"
        assert lr > 0, "lr must be positive"
        assert 0 <= epsilon_min <= epsilon <= 1, "Invalid epsilon range"
        assert memory_size > 0, "memory_size must be positive"
        assert batch_size > 0, "batch_size must be positive"
        assert target_update_freq > 0, "target_update_freq must be positive"
        assert 0 <= polyak_tau <= 1, "polyak_tau must be in [0, 1]"
        assert grad_clip_norm > 0, "grad_clip_norm must be positive"
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_dueling = use_dueling
        self.use_double_dqn = use_double_dqn
        self.polyak_tau = polyak_tau
        self.grad_clip_norm = grad_clip_norm
        
        # Set up device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Initialize networks
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, config.HIDDEN_DIM)
            self.target_net = DuelingDQN(state_dim, action_dim, config.HIDDEN_DIM)
        else:
            self.policy_net = DQN(state_dim, action_dim, config.HIDDEN_DIM)
            self.target_net = DQN(state_dim, action_dim, config.HIDDEN_DIM)
        
        # Move networks to device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Copy policy to target network
        self.update_target(hard_update=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.total_loss = 0.0
        self.losses = []
        self.best_lap_time = float('inf')
    
    def _greedy_action(self, state: np.ndarray) -> int:
        """Select the action with the highest Q-value (no exploration).
        
        Args:
            state: numpy array - current state
            
        Returns:
            int: action with highest Q-value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: numpy array - current state
            explore: bool - whether to use exploration (epsilon-greedy).
                     If False, always selects the greedy action.
        
        Returns:
            int: selected action
        """
        if not explore:
            return self._greedy_action(state)
        
        if torch.rand(1).item() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.action_dim)
        else:
            # Greedy action (exploitation)
            return self._greedy_action(state)
    
    def update(self) -> Optional[float]:
        """
        Update the policy network using a batch from the replay buffer.
        
        Returns:
            Optional[float]: Loss value if training occurred, None otherwise.
                           Returns None if there are not enough samples in the replay buffer
                           or if the sample returned None.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        if states is None:
            return None
        
        # Convert to PyTorch tensors on the correct device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate Q-values for current states
        current_q_values = self.policy_net(states)
        
        # Calculate Q-values for next states using target network
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy network to select action, target network for value
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states)
                next_q_values = next_q_values.gather(1, next_actions)
            else:
                # Standard DQN: use target network for both
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
        
        # Calculate target Q-values
        # target = reward + gamma * Q(s', a') * (1 - done)
        # Ensure all tensors have compatible shapes for broadcasting
        rewards = rewards.unsqueeze(1)  # Shape: [batch_size, 1]
        dones = dones.unsqueeze(1)  # Shape: [batch_size, 1]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Select Q-values for the actions taken
        current_q_values = current_q_values.gather(1, actions)
        
        # Calculate loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Update statistics
        self.total_loss += loss.item()
        self.losses.append(loss.item())
        
        # Log training info periodically
        if self.steps % 100 == 0:
            self.logger.debug(f"Step {self.steps}: Loss = {loss.item():.4f}, Epsilon = {self.epsilon:.4f}")
        
        # Decay epsilon and increment step counter (only if training occurred)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target()
        
        return loss.item()
    
    def update_target(self, hard_update: bool = False):
        """Update target network weights from policy network.
        
        Uses Polyak averaging (soft update) by default, or hard copy if specified.
        
        Args:
            hard_update: bool - if True, performs a hard copy of weights.
                         If False (default), uses Polyak averaging for smooth transition.
        """
        if hard_update or self.polyak_tau == 0:
            # Hard update: copy all weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # Polyak averaging: smooth interpolation of weights
            for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.polyak_tau * param.data + (1 - self.polyak_tau) * target_param.data)
        self.target_net.eval()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: numpy array - current state
            action: int - action taken
            reward: float - reward received
            next_state: numpy array - next state
            done: bool - whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            path: str - file path to save to
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'total_loss': self.total_loss,
            'best_lap_time': self.best_lap_time,
            'polyak_tau': self.polyak_tau,
            'grad_clip_norm': self.grad_clip_norm,
        }, path)
        self.logger.info(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: str - file path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.total_loss = checkpoint['total_loss']
        self.best_lap_time = checkpoint.get('best_lap_time', float('inf'))
        self.polyak_tau = checkpoint.get('polyak_tau', 0.005)
        self.grad_clip_norm = checkpoint.get('grad_clip_norm', 1.0)
        
        # Move networks to device after loading
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
        self.logger.info(f"Agent loaded from {path}")
    
    def increment_episode(self) -> None:
        """Increment the episode counter."""
        self.episodes += 1
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get training statistics.
        
        Returns:
            Dict[str, Union[int, float]]: Dictionary containing training statistics:
                - episodes: int - number of episodes completed
                - steps: int - number of training steps
                - epsilon: float - current exploration rate
                - avg_loss: float - average loss over last 100 updates
                - total_loss: float - cumulative loss
                - memory_size: int - current size of replay buffer
                - best_lap_time: float - best lap time achieved
        """
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'total_loss': self.total_loss,
            'memory_size': len(self.memory),
            'best_lap_time': self.best_lap_time,
        }


class RandomAgent:
    """
    Random agent for baseline comparison.
    Selects actions randomly without learning.
    """
    
    def __init__(self, action_dim: int = config.ACTION_DIM):
        """
        Initialize the random agent.
        
        Args:
            action_dim: int - number of actions
        """
        assert action_dim > 0, "action_dim must be positive"
        self.action_dim = action_dim
        self.episodes = 0
        self.steps = 0
    
    def select_action(self, state: Optional[np.ndarray] = None, explore: bool = True) -> int:
        """
        Select a random action.
        
        Args:
            state: Optional[np.ndarray] - ignored (present for interface consistency)
            explore: bool - ignored (present for interface consistency)
        
        Returns:
            int: random action
        """
        return np.random.randint(0, self.action_dim)
    
    def remember(self, *args, **kwargs) -> None:
        """No memory for random agent."""
        pass
    
    def update(self) -> None:
        """No learning for random agent."""
        pass
    
    def increment_episode(self) -> None:
        """Increment the episode counter."""
        self.episodes += 1
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics for the random agent.
        
        Returns:
            Dict[str, Union[int, float]]: Dictionary containing statistics
        """
        return {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': 1.0,  # Always exploring
            'avg_loss': 0.0,
            'total_loss': 0.0,
            'memory_size': 0,
        }


def test_agent():
    """Test the DQN agent and its variants."""
    print("Testing DQNAgent...")
    
    # Create agent
    agent = DQNAgent(state_dim=10, action_dim=5, batch_size=32)
    
    # Test action selection
    state = np.random.randn(10)
    
    # Test exploration
    action = agent.select_action(state, explore=True)
    print(f"Selected action (explore=True): {action}")
    assert 0 <= action < 5, "Invalid action"
    
    # Test exploitation (epsilon=0)
    agent.epsilon = 0.0
    action = agent.select_action(state, explore=True)
    print(f"Selected action (epsilon=0): {action}")
    assert 0 <= action < 5, "Invalid action"
    
    # Test non-explore mode always returns greedy action
    greedy_action = agent.select_action(state, explore=False)
    print(f"Selected action (explore=False): {greedy_action}")
    assert 0 <= greedy_action < 5, "Invalid action"
    
    # Test memory
    agent.remember(state, action, 1.0, state, False)
    assert len(agent.memory) == 1, "Memory should have 1 transition"
    
    # Test update (should return None with only 1 transition)
    loss = agent.update()
    print(f"Update loss (expected None): {loss}")
    assert loss is None, "Should return None when not enough samples"
    
    # Add more transitions
    for _ in range(31):
        agent.remember(state, action, 1.0, state, False)
    
    # Now update should work
    loss = agent.update()
    print(f"Update loss (should be > 0): {loss}")
    assert loss is not None and loss >= 0, "Loss should be non-negative"
    
    # Test update returns None when sample returns None
    # (This is harder to test directly, but the logic is in place)
    
    # Test Double DQN variant
    print("\nTesting Double DQN variant...")
    agent_ddqn = DQNAgent(state_dim=10, action_dim=5, batch_size=32, use_double_dqn=True)
    for _ in range(32):
        agent_ddqn.remember(state, action, 1.0, state, False)
    loss_ddqn = agent_ddqn.update()
    print(f"Double DQN update loss: {loss_ddqn}")
    assert loss_ddqn is not None and loss_ddqn >= 0, "Double DQN should work"
    
    # Test Dueling DQN variant
    print("\nTesting Dueling DQN variant...")
    agent_duel = DQNAgent(state_dim=10, action_dim=5, batch_size=32, use_dueling=True)
    for _ in range(32):
        agent_duel.remember(state, action, 1.0, state, False)
    loss_duel = agent_duel.update()
    print(f"Dueling DQN update loss: {loss_duel}")
    assert loss_duel is not None and loss_duel >= 0, "Dueling DQN should work"
    
    # Test Polyak averaging
    print("\nTesting Polyak averaging...")
    agent_polyak = DQNAgent(state_dim=10, action_dim=5, batch_size=32, polyak_tau=0.01)
    
    # First, modify the policy network weights (by doing a dummy update)
    # This ensures the policy weights differ from the target weights
    for _ in range(32):
        agent_polyak.remember(state, 0, 0.1, state, False)
    agent_polyak.update()  # This will modify policy_net weights
    
    # Now save the current policy network weights
    policy_weights = [p.clone() for p in agent_polyak.policy_net.parameters()]
    
    # Perform a hard update to set target to current policy
    agent_polyak.update_target(hard_update=True)
    
    # Now save the target network weights (which should equal policy weights)
    target_weights = [p.clone() for p in agent_polyak.target_net.parameters()]
    
    # Verify hard update copied weights exactly
    for p, policy_p in zip(target_weights, policy_weights):
        assert torch.equal(p, policy_p), "Hard update should copy weights exactly"
    
    # Now modify policy weights again and do a Polyak update
    for _ in range(32):
        agent_polyak.remember(state, 0, 0.1, state, False)
    agent_polyak.update()  # This will modify policy_net weights again
    
    # Save the new policy weights
    new_policy_weights = [p.clone() for p in agent_polyak.policy_net.parameters()]
    
    # Perform a Polyak update (soft update)
    agent_polyak.update_target()  # Should use soft update
    
    # Verify weights have changed from the previous target weights
    weights_changed = False
    for p, target_p in zip(agent_polyak.target_net.parameters(), target_weights):
        if not torch.equal(p, target_p):
            weights_changed = True
            break
    assert weights_changed, "Polyak update should modify weights"
    
    # Verify the new target weights are a blend of old target and new policy
    # With Polyak averaging: new_target = tau * policy + (1 - tau) * old_target
    for p, new_policy_p, old_target_p in zip(
        agent_polyak.target_net.parameters(), new_policy_weights, target_weights
    ):
        expected = 0.01 * new_policy_p + 0.99 * old_target_p
        assert torch.allclose(p, expected, rtol=1e-5), "Polyak update weights should be correct blend"
    
    # Test save/load
    import tempfile
    import os
    
    print("\nTesting save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_agent.pth")
        agent.save(path)
        
        # Create new agent and load
        agent2 = DQNAgent(state_dim=10, action_dim=5)
        agent2.load(path)
        
        assert agent2.epsilon == agent.epsilon, "Epsilon should match"
        assert agent2.steps == agent.steps, "Steps should match"
        assert agent2.best_lap_time == agent.best_lap_time, "Best lap time should match"
        
    # Test get_stats
    print("\nTesting get_stats...")
    stats = agent.get_stats()
    assert 'episodes' in stats, "Stats should include episodes"
    assert 'steps' in stats, "Stats should include steps"
    assert 'epsilon' in stats, "Stats should include epsilon"
    assert 'avg_loss' in stats, "Stats should include avg_loss"
    assert 'best_lap_time' in stats, "Stats should include best_lap_time"
    print(f"Stats: {stats}")
    
    # Test RandomAgent
    print("\nTesting RandomAgent...")
    random_agent = RandomAgent(action_dim=5)
    random_action = random_agent.select_action()
    print(f"Random agent action: {random_action}")
    assert 0 <= random_action < 5, "Random agent action out of range"
    
    random_stats = random_agent.get_stats()
    assert 'episodes' in random_stats, "Random agent stats should include episodes"
    print(f"Random agent stats: {random_stats}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_agent()
