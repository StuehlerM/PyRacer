"""DQN agent: action selection, the replay-driven update step, and save/load.

The agent is built with learning_starts=0 and a tiny batch so update() can be
exercised quickly without a long warmup.
"""
import numpy as np

from rl.agent import DQNAgent, RandomAgent
from utils.config import config

S = config.STATE_DIM
A = config.ACTION_DIM


def _make_agent(**overrides):
    params = dict(state_dim=S, action_dim=A, learning_starts=0,
                  batch_size=8, memory_size=500)
    params.update(overrides)
    return DQNAgent(**params)


def _fill_memory(agent, n):
    for _ in range(n):
        agent.remember(np.random.rand(S), np.random.randint(A),
                       float(np.random.rand()), np.random.rand(S),
                       bool(np.random.rand() > 0.5))


def test_greedy_action_in_range():
    agent = _make_agent()
    action = agent.select_action(np.random.rand(S).astype(np.float32), explore=False)
    assert 0 <= action < A


def test_update_returns_float_loss_once_ready():
    agent = _make_agent()
    _fill_memory(agent, 50)
    loss = agent.update()
    assert loss is not None
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_update_returns_none_without_enough_samples():
    agent = _make_agent(batch_size=64)
    _fill_memory(agent, 5)
    assert agent.update() is None


def test_on_env_step_decays_epsilon():
    agent = _make_agent()
    start = agent.epsilon
    agent.on_env_step()
    assert agent.epsilon <= start
    assert agent.epsilon >= agent.epsilon_min


def test_epsilon_does_not_decay_during_learning_warmup():
    agent = _make_agent(learning_starts=5)
    start = agent.epsilon

    for _ in range(agent.learning_starts):
        agent.on_env_step()

    assert agent.epsilon == start
    agent.on_env_step()
    assert agent.epsilon < start


def test_target_network_starts_synced_with_policy():
    agent = _make_agent()
    for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert (p == t).all()


def test_save_and_load_roundtrip(tmp_path):
    agent = _make_agent()
    path = tmp_path / "model.pth"
    agent.save(str(path))
    assert path.exists()
    # Loading into a fresh agent must not raise and must restore weights.
    other = _make_agent()
    other.load(str(path))
    for a, b in zip(agent.policy_net.parameters(), other.policy_net.parameters()):
        assert (a == b).all()


def test_random_agent_actions_in_range():
    ra = RandomAgent(action_dim=A)
    for _ in range(20):
        assert 0 <= ra.select_action() < A
