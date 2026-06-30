"""Replay buffer behaviour: capacity wrap-around, sample shapes, prioritization."""
import numpy as np

from rl.memory import ReplayBuffer, PrioritizedReplayBuffer
from utils.config import config

S = config.STATE_DIM


def _push_random(buf, n):
    for _ in range(n):
        buf.push(np.random.rand(S), np.random.randint(config.ACTION_DIM),
                 float(np.random.rand()), np.random.rand(S), bool(np.random.rand() > 0.5))


def test_length_caps_at_capacity():
    buf = ReplayBuffer(capacity=5, state_dim=S)
    assert len(buf) == 0
    _push_random(buf, 7)
    assert len(buf) == 5  # circular buffer overwrote the two oldest


def test_sample_returns_correct_shapes():
    buf = ReplayBuffer(capacity=100, state_dim=S)
    _push_random(buf, 30)
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, S)
    assert next_states.shape == (8, S)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)


def test_sample_none_when_not_enough_data():
    buf = ReplayBuffer(capacity=100, state_dim=S)
    _push_random(buf, 3)
    assert buf.sample(8) is None


def test_clear_empties_buffer():
    buf = ReplayBuffer(capacity=100, state_dim=S)
    _push_random(buf, 10)
    buf.clear()
    assert len(buf) == 0


def test_prioritized_sample_yields_indices_and_weights():
    buf = PrioritizedReplayBuffer(capacity=100, state_dim=S)
    _push_random(buf, 30)
    states, actions, rewards, next_states, dones, indices, weights = buf.sample(8)
    assert states.shape == (8, S)
    assert len(indices) == 8
    assert weights.shape == (8,)
    assert np.all(weights > 0)
    # Updating priorities must not raise and must keep them strictly positive.
    buf.update_priorities(indices, np.zeros(8))
    assert np.all(buf.priorities[indices] > 0)
