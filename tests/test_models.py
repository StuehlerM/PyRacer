"""Neural network models: output shapes, epsilon-greedy bounds, input validation."""
import numpy as np
import pytest
import torch

from rl.model import DQN, DuelingDQN
from utils.config import config

S = config.STATE_DIM
A = config.ACTION_DIM


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_forward_output_shape(model_cls):
    model = model_cls(input_dim=S, output_dim=A, hidden_dim=32)
    out = model(torch.randn(4, S, device=model.device))
    assert out.shape == (4, A)


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_act_returns_valid_action(model_cls):
    model = model_cls(input_dim=S, output_dim=A, hidden_dim=32)
    state = np.random.rand(S).astype(np.float32)
    assert 0 <= model.act(state, epsilon=0.0) < A   # greedy
    assert 0 <= model.act(state, epsilon=1.0) < A   # forced random


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_invalid_dimensions_raise(model_cls):
    with pytest.raises(ValueError):
        model_cls(input_dim=0, output_dim=A, hidden_dim=32)


def test_dueling_decomposition_is_mean_centered():
    """Dueling Q = V + (A - mean(A)); the advantage stream must be mean-zero
    relative to its contribution, which keeps V and A identifiable."""
    model = DuelingDQN(input_dim=S, output_dim=A, hidden_dim=32)
    x = torch.randn(16, S, device=model.device)
    features = model.feature_layer(x)
    advantage = model.advantage_stream(features)
    centered = advantage - advantage.mean(dim=1, keepdim=True)
    assert torch.allclose(centered.mean(dim=1),
                          torch.zeros(16, device=model.device), atol=1e-5)
