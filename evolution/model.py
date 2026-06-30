"""
Policy network for Neuroevolution.

Unlike the DQN/JEPA networks, this network is **never trained by backprop**.
Its weights are treated as a flat "genome" vector that a genetic algorithm
mutates and recombines. The network is a pure function:

    state -> action

KEY DIFFERENCE FROM DQN's network:
    DQN:        state -> Q-values, trained by gradient descent on a TD loss.
    Evolution:  state -> action logits, weights set directly from a genome.
                There is no optimizer and no loss — selection does the "learning".

Because there is no backprop, every forward pass runs under ``torch.no_grad()``
and the parameters have ``requires_grad=False``. We use ``tanh`` activations:
they are smooth and bounded, which keeps behavior stable under random weight
perturbations (large weights saturate instead of exploding).
"""

import numpy as np
import torch
import torch.nn as nn

from utils.config import config


class PolicyNetwork(nn.Module):
    """
    Feed-forward policy mapping a state to discrete action logits.

    The network exposes its weights as a single flat NumPy vector (the "genome")
    so the genetic algorithm can treat the whole policy as a point in parameter
    space to mutate and recombine.

    Attributes:
        state_dim: Dimension of the input state.
        action_dim: Number of discrete actions.
        hidden_dim: Width of the two hidden layers.
    """

    def __init__(self, state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM,
                 hidden_dim=config.EVO_HIDDEN_DIM):
        """
        Args:
            state_dim: Dimension of input state.
            action_dim: Number of discrete actions (output logits).
            hidden_dim: Hidden layer width.

        Raises:
            ValueError: If any dimension is not a positive integer.
        """
        super().__init__()

        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim, action_dim and hidden_dim must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compact MLP. tanh keeps activations bounded under random mutation.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # No gradients are ever needed: the GA sets weights directly.
        for p in self.parameters():
            p.requires_grad = False

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of states to action logits.

        Args:
            x: (batch_size, state_dim) state tensor.

        Returns:
            (batch_size, action_dim) action logits.
        """
        return self.net(x)

    @torch.no_grad()
    def act(self, state) -> int:
        """
        Select the greedy action for a single state.

        Neuroevolution has no exploration noise of its own — exploration happens
        across the *population*, not within a single policy — so this is a plain
        argmax over the action logits.

        Args:
            state: 1-D array-like of length ``state_dim``.

        Returns:
            int: chosen action in ``[0, action_dim)``.
        """
        state_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        logits = self.net(state_t)
        return int(torch.argmax(logits, dim=1).item())

    # ------------------------------------------------------------------
    # Genome <-> weights conversion (the bridge to the genetic algorithm)
    # ------------------------------------------------------------------
    def num_params(self) -> int:
        """Total number of scalar weights — the length of the genome vector."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def get_flat_params(self) -> np.ndarray:
        """
        Flatten all weights into one 1-D float32 vector (the genome).

        Returns:
            np.ndarray of shape ``(num_params,)``.
        """
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self.parameters()
        ]).astype(np.float32)

    @torch.no_grad()
    def set_flat_params(self, flat: np.ndarray) -> None:
        """
        Load a flat genome vector back into the network weights.

        Args:
            flat: 1-D array of length ``num_params``.

        Raises:
            ValueError: If ``flat`` has the wrong length.
        """
        flat = np.asarray(flat, dtype=np.float32).ravel()
        expected = self.num_params()
        if flat.size != expected:
            raise ValueError(
                f"Genome length {flat.size} does not match parameter count {expected}"
            )

        offset = 0
        for p in self.parameters():
            n = p.numel()
            chunk = flat[offset:offset + n].reshape(p.shape)
            p.copy_(torch.as_tensor(chunk, dtype=p.dtype, device=p.device))
            offset += n
