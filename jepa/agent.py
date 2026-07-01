"""
JEPA Agent: Self-supervised world model with energy-based planning.

This is the core of the JEPA approach. Unlike DQN which learns
"what action gives the most reward", JEPA:

1. Learns HOW THE WORLD WORKS (world model) — given state + action, what happens?
2. Plans toward GOALS — finds action sequences that reach desired states

No reward function needed. The agent learns purely from prediction.

COMPARISON TO DQN:
┌─────────────────────────────────────────────────────────────────┐
│ DQN Training Loop:                                              │
│   1. Observe state                                              │
│   2. Pick action (ε-greedy on Q-values)                         │
│   3. Get reward from environment                                │
│   4. Update Q-values toward: reward + γ * max_Q(next_state)     │
│                                                                 │
│ JEPA Training Loop:                                             │
│   1. Observe state                                              │
│   2. Pick action (CEM planning or random during warmup)         │
│   3. Observe next_state (NO reward needed!)                     │
│   4. Update world model: predict(encode(state), action) ≈       │
│      target_encode(next_state)                                  │
└─────────────────────────────────────────────────────────────────┘

PLANNING (action selection):
    Uses Cross-Entropy Method (CEM) to find good action sequences:
    1. Generate N random action sequences of length H
    2. For each sequence, simulate forward using world model
    3. Score each: how close does final predicted state get to goal?
    4. Keep top-K sequences, resample around their distribution
    5. Repeat for a few iterations
    6. Execute first action of best sequence

This is Model Predictive Control (MPC) — re-plan every step.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.config import config
from .model import (
    StateEncoder, Predictor, create_target_encoder, 
    update_target_encoder, vicreg_loss
)
from .memory import TransitionBuffer, GoalBuffer


class JEPAAgent:
    """
    JEPA Agent with self-supervised world model and CEM planning.
    
    Provides the same interface as DQNAgent for compatibility:
        - select_action(state, explore)
        - remember(state, action, reward, next_state, done)
        - update()
        - save(path) / load(path)
    
    But internally works completely differently:
        - No Q-values, no reward processing
        - Learns world dynamics in latent space
        - Plans actions via search (CEM)
    """
    
    def __init__(
        self,
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        latent_dim=config.JEPA_LATENT_DIM,
        hidden_dim=config.JEPA_HIDDEN_DIM,
        encoder_lr=config.JEPA_ENCODER_LR,
        predictor_lr=config.JEPA_PREDICTOR_LR,
        ema_tau=config.JEPA_EMA_TAU,
        planning_horizon=config.JEPA_PLANNING_HORIZON,
        cem_candidates=config.JEPA_CEM_CANDIDATES,
        cem_elites=config.JEPA_CEM_ELITES,
        cem_iterations=config.JEPA_CEM_ITERATIONS,
        warmup_steps=config.JEPA_WARMUP_STEPS,
        batch_size=config.JEPA_BATCH_SIZE,
        memory_size=config.JEPA_MEMORY_SIZE,
        goal_buffer_size=config.JEPA_GOAL_BUFFER_SIZE,
        goal_progress_threshold=config.JEPA_GOAL_PROGRESS_THRESHOLD,
        train_freq=config.JEPA_TRAIN_FREQ,
        **kwargs,  # Accept and ignore RL-specific params for interface compat
    ):
        """
        Initialize JEPA agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            latent_dim: Dimension of latent representation
            hidden_dim: Hidden layer size
            encoder_lr: Learning rate for encoder
            predictor_lr: Learning rate for predictor
            ema_tau: EMA rate for target encoder
            planning_horizon: Steps to plan ahead
            cem_candidates: Action sequences per CEM iteration
            cem_elites: Top sequences kept per iteration
            cem_iterations: CEM refinement rounds
            warmup_steps: Random actions before planning starts
            batch_size: Training batch size
            memory_size: Transition buffer capacity
            goal_buffer_size: Goal buffer capacity
            goal_progress_threshold: Min progress for goal states
            train_freq: Train world model every N env steps
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau
        self.planning_horizon = planning_horizon
        self.cem_candidates = cem_candidates
        self.cem_elites = cem_elites
        self.cem_iterations = cem_iterations
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.goal_progress_threshold = goal_progress_threshold
        self.train_freq = train_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # === Networks ===
        # Online encoder (trained via backprop)
        self.encoder = StateEncoder(state_dim, latent_dim, hidden_dim).to(self.device)
        # Target encoder (updated via EMA — provides stable prediction targets)
        self.target_encoder = create_target_encoder(self.encoder).to(self.device)
        # Predictor (trained to predict target_encoder output)
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim).to(self.device)
        
        # === Optimizer ===
        # Joint optimization of encoder + predictor
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.predictor.parameters(), 'lr': predictor_lr},
        ])
        
        # === Memory ===
        self.memory = TransitionBuffer(memory_size, state_dim)
        self.goal_buffer = GoalBuffer(goal_buffer_size, latent_dim)
        
        # === State tracking ===
        self.env_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self._last_progress = 0.0
        
        # For interface compatibility with DQN (epsilon not used in JEPA)
        self.epsilon = 1.0  # Starts at 1 (all random), drops to 0 after warmup
        
        # Training stats
        self._last_loss = {}
    
    def select_action(self, state, explore=True):
        """
        Select action using CEM planning (or random during warmup).
        
        During warmup (first N steps):
            Random actions to build world model data.
            
        After warmup:
            CEM planning through the learned world model.
            Find action sequence that moves latent state toward goal.
        
        Args:
            state: numpy array of current state
            explore: If True and during warmup, use random actions
            
        Returns:
            int: selected action (0 to action_dim-1)
        """
        # Phase 1: Random exploration during warmup
        if explore and self.env_steps < self.warmup_steps:
            return np.random.randint(0, self.action_dim)
        
        # Phase 2: CEM planning using world model
        # If no goals yet, fall back to random
        if len(self.goal_buffer) == 0:
            return np.random.randint(0, self.action_dim)
        
        return self._cem_plan(state)
    
    @torch.no_grad()
    def _cem_plan(self, state):
        """
        Cross-Entropy Method planning through the world model.
        
        Algorithm:
            1. Initialize action distribution (uniform over all actions)
            2. For each CEM iteration:
                a. Sample N action sequences from distribution
                b. Simulate each sequence through world model
                c. Score: distance from final predicted state to goal
                d. Keep top-K (elite) sequences
                e. Update distribution based on elites
            3. Return first action of best sequence
        
        This is a form of Model Predictive Control (MPC):
            - Plan ahead using the learned model
            - Execute only the first action
            - Re-plan at next timestep (handles model errors)
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            int: Best first action
        """
        self.encoder.eval()
        self.predictor.eval()
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get current latent state
        z_current = self.encoder(state_tensor)  # (1, latent_dim)
        
        # Get goal latent state (mean of goal buffer)
        goal_np = self.goal_buffer.get_mean_goal()
        z_goal = torch.as_tensor(goal_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Initialize action probabilities (uniform)
        # Shape: (horizon,) array of probability distributions over actions
        action_probs = np.ones((self.planning_horizon, self.action_dim)) / self.action_dim
        
        best_sequence = None
        best_score = float('inf')
        
        for cem_iter in range(self.cem_iterations):
            # Sample N action sequences from current distribution
            sequences = np.zeros((self.cem_candidates, self.planning_horizon), dtype=np.int64)
            for t in range(self.planning_horizon):
                sequences[:, t] = np.random.choice(
                    self.action_dim, size=self.cem_candidates, p=action_probs[t]
                )
            
            # Simulate each sequence through world model
            scores = np.zeros(self.cem_candidates)
            
            for i in range(self.cem_candidates):
                z = z_current.clone()  # Start from current state
                
                for t in range(self.planning_horizon):
                    action_t = torch.tensor([sequences[i, t]], device=self.device)
                    z = self.predictor(z, action_t)  # Predict next latent state
                
                # Score: how close is the final predicted state to the goal?
                # Lower distance = better score
                distance = F.mse_loss(z, z_goal).item()
                scores[i] = distance
            
            # Keep elite sequences (lowest distance to goal)
            elite_indices = np.argsort(scores)[:self.cem_elites]
            elite_sequences = sequences[elite_indices]
            
            # Track best overall
            if scores[elite_indices[0]] < best_score:
                best_score = scores[elite_indices[0]]
                best_sequence = elite_sequences[0]
            
            # Update action distribution based on elites
            for t in range(self.planning_horizon):
                counts = np.bincount(elite_sequences[:, t], minlength=self.action_dim)
                # Smooth to avoid zero probabilities
                action_probs[t] = (counts + 0.1) / (counts + 0.1).sum()
        
        self.encoder.train()
        self.predictor.train()
        
        # Return first action of best sequence
        return int(best_sequence[0]) if best_sequence is not None else 0
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store transition for world model training.
        
        Note: reward and done are accepted for interface compatibility
        but JEPA doesn't use them for learning! The world model learns
        purely from (state, action, next_state) prediction.
        
        We do track progress (part of state) to identify goal states.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward (ignored by JEPA — kept for interface compat)
            next_state: Next state
            done: Episode done flag (ignored for learning)
        """
        # Extract progress from state (last element in our state vector)
        progress = float(state[-1]) if len(state) > 0 else 0.0
        
        self.memory.push(state, action, next_state, progress)
        self.env_steps += 1
        self._last_progress = progress
    
    def update(self):
        """
        Train the world model (encoder + predictor).
        
        JEPA training objective:
            predictor(encoder(state), action) ≈ target_encoder(next_state)
            
        With VICReg regularization to prevent collapse.
        
        Compare to DQN update:
            DQN minimizes: (Q(s,a) - (reward + γ * max Q(s',a')))²
            JEPA minimizes: ||predict(encode(s), a) - target_encode(s')||²
                            + variance reg + covariance reg
        
        Returns:
            float: Total loss value, or None if not enough data
        """
        # Only train periodically and after minimum data collected
        if self.env_steps < self.batch_size:
            return None
        if self.env_steps % self.train_freq != 0:
            return None
        
        # Sample batch from transition buffer
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, next_states = batch
        
        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        
        # === Forward pass ===
        # Online encoder encodes current state
        z = self.encoder(states_t)
        
        # Predictor predicts next latent state
        z_pred = self.predictor(z, actions_t)
        
        # Target encoder encodes actual next state (no gradient!)
        with torch.no_grad():
            z_target = self.target_encoder(next_states_t)
        
        # === Compute VICReg loss ===
        loss_dict = vicreg_loss(z_pred, z_target)
        total_loss = loss_dict['total']
        
        # === Backprop ===
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        
        # === Update target encoder (EMA) ===
        update_target_encoder(self.encoder, self.target_encoder, self.ema_tau)
        
        # === Update goal buffer ===
        self._update_goals()
        
        self.train_steps += 1
        self._last_loss = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        # Update epsilon for display compatibility (shows warmup progress)
        if self.env_steps < self.warmup_steps:
            self.epsilon = 1.0 - (self.env_steps / self.warmup_steps) * 0.9
        else:
            self.epsilon = 0.1
        
        return total_loss.item()
    
    @torch.no_grad()
    def _update_goals(self):
        """
        Populate goal buffer with latent encodings of high-progress states.
        
        This is how JEPA defines "what's good" without reward:
            States where the car has made significant track progress
            are encoded into latent space and stored as goals.
            
        The planner then searches for action sequences that move
        the predicted latent state toward these goal representations.
        """
        high_progress_states = self.memory.sample_high_progress(
            min(32, max(1, len(self.memory) // 10)),
            threshold=self.goal_progress_threshold
        )
        
        if high_progress_states is None:
            return
        
        states_t = torch.as_tensor(high_progress_states, dtype=torch.float32, device=self.device)
        # Use target encoder for stable goal representations
        z_goals = self.target_encoder(states_t).cpu().numpy()
        self.goal_buffer.push_batch(z_goals)
    
    def on_env_step(self):
        """Called after each environment step (for interface compatibility)."""
        pass
    
    def increment_episode(self):
        """Called at end of each episode."""
        self.episodes += 1
    
    def get_stats(self):
        """
        Get training statistics.
        
        Returns:
            dict with training info
        """
        return {
            'env_steps': self.env_steps,
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'goal_buffer_size': len(self.goal_buffer),
            'memory_size': len(self.memory),
            'warmup_complete': self.env_steps >= self.warmup_steps,
            'loss': self._last_loss,
        }
    
    def save(self, path):
        """
        Save JEPA model.
        
        Args:
            path: File path to save to
        """
        torch.save({
            'encoder': self.encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'env_steps': self.env_steps,
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'latent_dim': self.latent_dim,
            }
        }, path)
    
    def load(self, path):
        """
        Load JEPA model.
        
        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'env_steps' in checkpoint:
            self.env_steps = checkpoint['env_steps']
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']
        if 'episodes' in checkpoint:
            self.episodes = checkpoint['episodes']
