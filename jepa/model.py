"""
JEPA (Joint Embedding Predictive Architecture) neural network models.

Unlike RL models that output Q-values or policies, JEPA models learn a
**world model** in latent space. The key insight:

    Instead of predicting raw next states (pixels/sensors), predict
    abstract *representations* of next states. This avoids wasting
    capacity on irrelevant details and learns useful structure.

Architecture:
    StateEncoder(fθ):  state → latent z
    Predictor(gφ):     (z_t, action) → predicted z_{t+1}
    TargetEncoder:     EMA copy of StateEncoder (stable targets)

Training signal:
    Minimize distance between:
        gφ(fθ(s_t), a_t)  and  target_encoder(s_{t+1})
    
    Plus VICReg regularization to prevent representational collapse
    (where all states map to the same point in latent space).

References:
    - LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
    - Bardes et al., "VICReg" (2022)
    - Assran et al., "I-JEPA" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from utils.config import config


class StateEncoder(nn.Module):
    """
    Encodes raw state (sensor readings) into a latent representation.
    
    This is the "perception" module — it takes the 11-dim sensor state
    and maps it to a 64-dim latent vector z that captures the essential
    structure of the driving situation.
    
    Key difference from DQN's network:
        DQN: state → Q-values (one per action)
        JEPA Encoder: state → latent embedding (abstract representation)
    """
    
    def __init__(self, state_dim=config.STATE_DIM, latent_dim=config.JEPA_LATENT_DIM,
                 hidden_dim=config.JEPA_HIDDEN_DIM):
        """
        Args:
            state_dim: Dimension of input state (default: 11)
            latent_dim: Dimension of latent representation (default: 64)
            hidden_dim: Hidden layer width (default: 128)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # MLP encoder with layer normalization for training stability
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state into latent representation.
        
        Args:
            state: (batch_size, state_dim) raw sensor state
            
        Returns:
            z: (batch_size, latent_dim) latent representation
        """
        return self.net(state)


class Predictor(nn.Module):
    """
    Predicts the next latent state given current latent state and action.
    
    This is the "world model" — it learns the dynamics of the environment
    entirely in latent space. Given where you are (z_t) and what you do
    (action), it predicts where you'll end up (z_{t+1}).
    
    Key insight:
        Predicting in latent space is easier than predicting raw states.
        The encoder filters out irrelevant information, so the predictor
        only needs to model the essential dynamics.
    
    The predictor is intentionally a separate network from the encoder.
    This asymmetry is crucial: the predictor can be simpler/smaller than
    the encoder, which helps prevent collapse (the encoder can't just
    learn the identity function).
    """
    
    def __init__(self, latent_dim=config.JEPA_LATENT_DIM, action_dim=config.ACTION_DIM,
                 hidden_dim=config.JEPA_HIDDEN_DIM):
        """
        Args:
            latent_dim: Dimension of latent space
            action_dim: Number of discrete actions (used for one-hot encoding)
            hidden_dim: Hidden layer width
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Input: concatenation of latent state z and one-hot action
        input_dim = latent_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state.
        
        Args:
            z: (batch_size, latent_dim) current latent state
            action: (batch_size,) integer actions OR (batch_size, action_dim) one-hot
            
        Returns:
            z_pred: (batch_size, latent_dim) predicted next latent state
        """
        # Convert integer actions to one-hot
        if action.dim() == 1 or (action.dim() == 2 and action.shape[1] == 1):
            action = action.view(-1).long()
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
        
        # Concatenate latent state with action
        x = torch.cat([z, action_onehot], dim=-1)
        return self.net(x)


def create_target_encoder(encoder: StateEncoder) -> StateEncoder:
    """
    Create a target encoder as a frozen copy of the online encoder.
    
    The target encoder uses Exponential Moving Average (EMA) of the
    online encoder's parameters. This provides stable prediction targets
    and is a key technique to prevent representational collapse.
    
    Why EMA works:
        - Without it, both encoder and predictor could collapse to trivial solutions
        - The slowly-moving target creates a "moving goalpost" that forces
          the online encoder to keep improving
        - Similar to the target network in DQN, but smoother (Polyak averaging)
    
    Args:
        encoder: The online StateEncoder to copy
        
    Returns:
        A frozen copy (no gradients) of the encoder
    """
    target = copy.deepcopy(encoder)
    # Freeze parameters — target encoder is updated only via EMA
    for param in target.parameters():
        param.requires_grad = False
    return target


@torch.no_grad()
def update_target_encoder(online: StateEncoder, target: StateEncoder, 
                          tau: float = config.JEPA_EMA_TAU):
    """
    Update target encoder parameters via exponential moving average.
    
    target_params = (1 - tau) * target_params + tau * online_params
    
    With tau=0.005, the target encoder changes very slowly, providing
    stable prediction targets while still tracking the online encoder.
    
    This is the same concept as Polyak averaging in DQN's target network,
    but here it's essential for preventing collapse rather than just
    stabilizing Q-learning.
    
    Args:
        online: Online encoder (actively being trained)
        target: Target encoder (EMA copy, provides stable targets)
        tau: Interpolation rate (small = slow change = more stable)
    """
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


def vicreg_loss(z_pred: torch.Tensor, z_target: torch.Tensor,
                lambda_var: float = config.JEPA_VICREG_LAMBDA,
                mu_inv: float = config.JEPA_VICREG_MU,
                nu_cov: float = config.JEPA_VICREG_NU) -> dict:
    """
    VICReg loss: Variance-Invariance-Covariance Regularization.
    
    This is the key to preventing representational collapse in JEPA.
    Without it, the encoder could map all states to the same point (collapse).
    
    Three terms:
    
    1. INVARIANCE (prediction accuracy):
       MSE between predicted z and target z.
       "The predictor should accurately predict the next state."
    
    2. VARIANCE (information preservation):
       Forces each latent dimension to have variance > threshold.
       "Don't let any dimension collapse to a constant."
    
    3. COVARIANCE (decorrelation):
       Penalizes correlation between different latent dimensions.
       "Each dimension should capture different information."
    
    Together, these ensure the latent space is:
        - Predictive (invariance term)
        - Information-rich (variance term)
        - Efficiently structured (covariance term)
    
    Args:
        z_pred: Predicted next latent states (batch_size, latent_dim)
        z_target: Target next latent states from EMA encoder (batch_size, latent_dim)
        lambda_var: Weight for variance term
        mu_inv: Weight for invariance term
        nu_cov: Weight for covariance term
        
    Returns:
        Dict with total loss and individual components
    """
    batch_size, latent_dim = z_pred.shape
    
    # === INVARIANCE LOSS ===
    # MSE between prediction and target (the core prediction objective)
    inv_loss = F.mse_loss(z_pred, z_target)
    
    # === VARIANCE LOSS ===
    # Force each dimension to maintain minimum variance across the batch
    # This prevents collapse: if variance → 0, the dim is useless
    std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    std_target = torch.sqrt(z_target.var(dim=0) + 1e-4)
    # Hinge loss: penalize only when std < 1
    var_loss = (F.relu(1.0 - std_pred).mean() + F.relu(1.0 - std_target).mean()) / 2
    
    # === COVARIANCE LOSS ===
    # Penalize off-diagonal elements of the covariance matrix
    # This decorrelates dimensions so each captures unique information
    z_pred_centered = z_pred - z_pred.mean(dim=0)
    z_target_centered = z_target - z_target.mean(dim=0)
    
    cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
    cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)
    
    # Zero out diagonal (we only penalize off-diagonal correlations)
    mask = ~torch.eye(latent_dim, dtype=torch.bool, device=z_pred.device)
    cov_loss = (cov_pred[mask].pow(2).sum() + cov_target[mask].pow(2).sum()) / (2 * latent_dim)
    
    # Total loss
    total_loss = mu_inv * inv_loss + lambda_var * var_loss + nu_cov * cov_loss
    
    return {
        'total': total_loss,
        'invariance': inv_loss,
        'variance': var_loss,
        'covariance': cov_loss,
    }
