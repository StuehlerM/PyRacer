"""
Memory buffers for JEPA training.

Key difference from RL replay buffers:
    RL buffer stores: (state, action, reward, next_state, done)
    JEPA buffer stores: (state, action, next_state)
    
    No reward! JEPA learns world dynamics without any reward signal.
    The agent figures out "what's good" through goal-directed planning,
    not through reward accumulation.

Also includes a GoalBuffer that stores latent representations of
"desirable" states. These serve as planning targets during inference.
"""

import numpy as np
from utils.config import config


class TransitionBuffer:
    """
    Buffer for JEPA world model training.
    
    Stores simple (state, action, next_state) transitions.
    No reward or done signal — JEPA doesn't need them for learning.
    
    The world model learns: "if I'm in state s and take action a,
    what state will I end up in?" This is pure dynamics prediction.
    
    Compare to RL:
        RL asks: "what action maximizes future reward?"
        JEPA asks: "what happens if I take this action?"
        
    JEPA then uses the learned dynamics for planning toward goals.
    """
    
    def __init__(self, capacity=config.JEPA_MEMORY_SIZE, state_dim=config.STATE_DIM):
        """
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vector
        """
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Pre-allocate arrays for efficient storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        
        # Also store progress for goal buffer population
        self.progress = np.zeros(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, next_state, progress=0.0):
        """
        Store a transition.
        
        Args:
            state: Current state (numpy array)
            action: Action taken (integer)
            next_state: Resulting state (numpy array)
            progress: Track progress at this state (for goal identification)
        """
        self.states[self.pos] = np.asarray(state, dtype=np.float32)
        self.actions[self.pos] = action
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32)
        self.progress[self.pos] = progress
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Random sample for world model training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, next_states) numpy arrays,
            or None if buffer has fewer samples than batch_size
        """
        if self.size < batch_size:
            return None
        
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
        )
    
    def sample_high_progress(self, batch_size, threshold=config.JEPA_GOAL_PROGRESS_THRESHOLD):
        """
        Sample states with high track progress (for goal buffer).
        
        These "successful" states become planning targets — the JEPA agent
        plans action sequences to reach latent states similar to these.
        
        Args:
            batch_size: Number of states to sample
            threshold: Minimum progress value to qualify
            
        Returns:
            States array or None if not enough qualifying states
        """
        mask = self.progress[:self.size] >= threshold
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) < batch_size:
            return None
        
        indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return self.states[indices]
    
    def __len__(self):
        return self.size


class GoalBuffer:
    """
    Stores latent representations of "goal" states for planning.
    
    In RL, the agent chases reward. In JEPA, the agent plans toward
    goal states in latent space. This buffer stores latent encodings
    of states that represent "good driving" (high progress, fast speed).
    
    During planning, the CEM planner searches for action sequences that
    would move the predicted latent state closer to these goal states.
    
    How goals are defined (no human reward engineering needed):
        1. Agent explores randomly during warmup
        2. States with high track progress are identified as "good"
        3. Their latent encodings become planning targets
        4. As the agent gets better, goals naturally improve
    
    This is fundamentally different from RL reward shaping:
        RL: Human designs reward function → agent maximizes it
        JEPA: Agent identifies successful states → plans to reach them
    """
    
    def __init__(self, capacity=config.JEPA_GOAL_BUFFER_SIZE, 
                 latent_dim=config.JEPA_LATENT_DIM):
        """
        Args:
            capacity: Maximum number of goal embeddings
            latent_dim: Dimension of latent space
        """
        self.capacity = capacity
        self.latent_dim = latent_dim
        
        self.goals = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, latent_state):
        """
        Add a goal latent state.
        
        Args:
            latent_state: (latent_dim,) numpy array of latent encoding
        """
        self.goals[self.pos] = np.asarray(latent_state, dtype=np.float32)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(self, latent_states):
        """
        Add multiple goal latent states.
        
        Args:
            latent_states: (N, latent_dim) numpy array
        """
        for z in latent_states:
            self.push(z)
    
    def sample(self, batch_size=1):
        """
        Sample goal states for planning.
        
        Args:
            batch_size: Number of goals to sample
            
        Returns:
            (batch_size, latent_dim) numpy array, or None if empty
        """
        if self.size == 0:
            return None
        
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.goals[indices]
    
    def get_mean_goal(self):
        """
        Get the average goal state (centroid of all goals).
        
        Using the mean provides a stable planning target that represents
        the "average good state" rather than any specific one.
        
        Returns:
            (latent_dim,) numpy array, or None if empty
        """
        if self.size == 0:
            return None
        return self.goals[:self.size].mean(axis=0)
    
    def __len__(self):
        return self.size
