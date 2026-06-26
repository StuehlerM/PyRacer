"""
Experience replay buffer for DQN training.
"""
import random
import numpy as np
from collections import deque, namedtuple


# Define a transition tuple for storing experiences
Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience replay buffer that stores transitions for RL training.
    
    Allows random sampling for training and maintains a fixed size.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: int - maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
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
        # Convert numpy arrays to bytes for storage
        state_bytes = state.astype(np.float32).tobytes()
        next_state_bytes = next_state.astype(np.float32).tobytes()
        
        transition = Transition(
            state=state_bytes,
            action=action,
            reward=reward,
            next_state=next_state_bytes,
            done=done
        )
        
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size: int - number of transitions to sample
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and convert back to numpy arrays
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        state_dim = None
        
        for transition in batch:
            # Convert state bytes back to numpy array
            if state_dim is None:
                state_dim = len(transition.state) // 4  # float32 is 4 bytes
            
            state = np.frombuffer(transition.state, dtype=np.float32).reshape(1, -1)
            next_state = np.frombuffer(transition.next_state, dtype=np.float32).reshape(1, -1)
            
            states.append(state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(next_state)
            dones.append(transition.done)
        
        # Stack into numpy arrays
        states = np.concatenate(states, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the number of transitions in the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
    
    def get_all(self):
        """
        Get all transitions in the buffer.
        
        Returns:
            list: all Transition tuples
        """
        return list(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples transitions based on their TD error (priority).
    This is more advanced and can improve learning efficiency.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: int - maximum number of transitions
            alpha: float - how much prioritization is used (0 = uniform, 1 = full)
            beta: float - importance sampling correction factor
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.pos = 0  # Current position in buffer
    
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
        # Convert numpy arrays to bytes for storage
        state_bytes = state.astype(np.float32).tobytes()
        next_state_bytes = next_state.astype(np.float32).tobytes()
        
        transition = Transition(
            state=state_bytes,
            action=action,
            reward=reward,
            next_state=next_state_bytes,
            done=done
        )
        
        # New transitions get maximum priority
        max_priority = max(self.priorities, default=1.0)
        
        # Store in buffer with stable indices
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for specific transitions.
        
        Args:
            indices: list - indices of transitions to update
            priorities: list - new priorities for those transitions
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions using prioritized sampling.
        
        Args:
            batch_size: int - number of transitions to sample
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, 
                                   p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Get sampled transitions
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            transition = self.buffer[idx]
            
            # Convert state bytes back to numpy array
            state_dim = len(transition.state) // 4
            state = np.frombuffer(transition.state, dtype=np.float32).reshape(1, -1)
            next_state = np.frombuffer(transition.next_state, dtype=np.float32).reshape(1, -1)
            
            states.append(state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(next_state)
            dones.append(transition.done)
        
        # Stack into numpy arrays
        states = np.concatenate(states, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def __len__(self):
        """Return the number of transitions in the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer = []
        self.priorities = []
        self.pos = 0
