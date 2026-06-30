"""Experience replay buffers for DQN training."""
import numpy as np
from collections import namedtuple


# Define a transition tuple for storing experiences
Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience replay buffer that stores transitions for RL training.
    
    Allows random sampling for training and maintains a fixed size.
    """
    
    def __init__(self, capacity, state_dim, action_dtype=np.int64):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: int - maximum number of transitions to store
            state_dim: int - number of values in each state
            action_dtype: numpy dtype - dtype for stored actions
        """
        # Fixed-capacity replay keeps memory bounded while still reusing old slots.
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: numpy array - current state
            action: int or float - action taken
            reward: float - reward received
            next_state: numpy array - next state
            done: bool - whether episode ended
        """
        # Replay buffer stores transitions out of order so training data is less temporally correlated.
        self.states[self.pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.pos] = done
        # Circular buffer wraps around and overwrites oldest experience when full.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size: int - number of transitions to sample
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            return None

        # Uniform random sampling turns sequential gameplay into i.i.d.-like minibatches.
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def __len__(self):
        """Return the number of transitions in the buffer."""
        return self.size
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.pos = 0
        self.size = 0
    
    def get_all(self):
        """
        Get all transitions in the buffer.
        
        Returns:
            list: all Transition tuples
        """
        return [
            Transition(
                self.states[i].copy(),
                self.actions[i].item() if hasattr(self.actions[i], "item") else self.actions[i],
                float(self.rewards[i]),
                self.next_states[i].copy(),
                bool(self.dones[i]),
            )
            for i in range(self.size)
        ]


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples transitions based on their TD error (priority).
    This is more advanced and can improve learning efficiency.
    """
    
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, action_dtype=np.int64):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: int - maximum number of transitions
            alpha: float - how much prioritization is used (0 = uniform, 1 = full)
            beta: float - importance sampling correction factor
            action_dtype: numpy dtype - dtype for stored actions
        """
        # Prioritized replay trades unbiased sampling for faster learning from surprising transitions.
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=action_dtype)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0  # Current position in buffer
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition with maximum priority (for new transitions).
        
        Args:
            state: numpy array - current state
            action: int or float - action taken
            reward: float - reward received
            next_state: numpy array - next state
            done: bool - whether episode ended
        """
        # New transitions start at max priority so agent sees fresh experience at least once.
        max_priority = float(self.priorities[:self.size].max()) if self.size > 0 else 1.0

        self.states[self.pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_priority
        # Same circular buffer idea: bounded memory, oldest items replaced first.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for specific transitions.
        
        Args:
            indices: list - indices of transitions to update
            priorities: list - new priorities for those transitions
        """
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                # Higher TD error means "more surprising", so replay will sample it more often.
                self.priorities[idx] = max(float(priority), 1e-6)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling.
        
        Args:
            batch_size: int - number of transitions to sample
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size < batch_size:
            return None
        
        # Alpha controls how strongly TD-error affects sampling probability.
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            probabilities = probabilities / probability_sum
        
        # Non-uniform sampling focuses compute on transitions likely to teach most.
        indices = np.random.choice(self.size, size=batch_size,
                                   p=probabilities, replace=False)
        
        # Importance weights partially undo bias introduced by prioritized, non-uniform sampling.
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        weights = weights.astype(np.float32)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )
    
    def __len__(self):
        """Return the number of transitions in the buffer."""
        return self.size
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.pos = 0
        self.size = 0
        self.priorities.fill(0.0)
