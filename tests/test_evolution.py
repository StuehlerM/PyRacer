"""Neuroevolution: policy network, GA population, and the EvolutionAgent.

These tests stay tiny (small populations, few generations) so the whole module
is exercised in a fraction of a second with no display required.
"""
import numpy as np
import pytest

from evolution.model import PolicyNetwork
from evolution.population import Population
from evolution.agent import EvolutionAgent
from utils.config import config

S = config.STATE_DIM
A = config.ACTION_DIM


# ----------------------------------------------------------------------
# PolicyNetwork
# ----------------------------------------------------------------------
def test_forward_output_shape():
    import torch
    net = PolicyNetwork(state_dim=S, action_dim=A, hidden_dim=16)
    out = net(torch.randn(4, S, device=net.device))
    assert out.shape == (4, A)


def test_act_returns_valid_action():
    net = PolicyNetwork(state_dim=S, action_dim=A, hidden_dim=16)
    state = np.random.rand(S).astype(np.float32)
    assert 0 <= net.act(state) < A


def test_flat_param_roundtrip():
    net = PolicyNetwork(state_dim=S, action_dim=A, hidden_dim=16)
    genome = net.get_flat_params()
    assert genome.shape == (net.num_params(),)
    assert genome.dtype == np.float32

    new = np.random.randn(net.num_params()).astype(np.float32)
    net.set_flat_params(new)
    assert np.allclose(net.get_flat_params(), new, atol=1e-5)


def test_set_flat_params_wrong_size_raises():
    net = PolicyNetwork(state_dim=S, action_dim=A, hidden_dim=16)
    with pytest.raises(ValueError):
        net.set_flat_params(np.zeros(net.num_params() + 1, dtype=np.float32))


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        PolicyNetwork(state_dim=0, action_dim=A, hidden_dim=16)


# ----------------------------------------------------------------------
# Population (genetic algorithm)
# ----------------------------------------------------------------------
def test_population_sizes_are_stable_across_generations():
    pop = Population(num_params=20, pop_size=10, seed=0)
    genomes = pop.ask()
    assert len(genomes) == 10
    assert all(g.shape == (20,) for g in genomes)

    pop.tell(np.random.rand(10))
    assert len(pop.ask()) == 10
    assert pop.generation == 1


def test_population_best_is_monotonic_non_decreasing():
    """Elitism guarantees the all-time best fitness never drops."""
    pop = Population(num_params=20, pop_size=12, seed=1)
    best_history = []
    for _ in range(8):
        pop.tell(np.random.rand(pop.pop_size))
        best_history.append(pop.best_fitness)
    assert all(b2 >= b1 for b1, b2 in zip(best_history, best_history[1:]))


def test_population_improves_on_convex_fitness():
    """On a simple concave objective (maximize -||g||^2), evolution should
    move the best genome closer to the optimum (the zero vector)."""
    pop = Population(num_params=30, pop_size=40, mutation_std=0.2,
                     init_std=0.5, seed=2)

    def fitness(g):
        return -float(np.sum(g ** 2))

    first_best = None
    for _ in range(25):
        scores = [fitness(g) for g in pop.ask()]
        if first_best is None:
            first_best = max(scores)
        pop.tell(scores)

    assert pop.best_fitness > first_best


def test_population_tell_validates_length():
    pop = Population(num_params=10, pop_size=6, seed=0)
    with pytest.raises(ValueError):
        pop.tell([0.0, 1.0])  # wrong number of fitnesses


def test_population_seed_is_reproducible():
    a = Population(num_params=15, pop_size=8, seed=123)
    b = Population(num_params=15, pop_size=8, seed=123)
    fits = np.linspace(0, 1, 8)
    a.tell(fits)
    b.tell(fits)
    for ga, gb in zip(a.ask(), b.ask()):
        assert np.allclose(ga, gb)


# ----------------------------------------------------------------------
# EvolutionAgent
# ----------------------------------------------------------------------
def _make_agent(**overrides):
    params = dict(state_dim=S, action_dim=A, hidden_dim=16, pop_size=8, seed=0)
    params.update(overrides)
    return EvolutionAgent(**params)


def test_agent_select_action_in_range():
    agent = _make_agent()
    action = agent.select_action(np.random.rand(S).astype(np.float32), explore=False)
    assert 0 <= action < A


def test_agent_noop_hooks_do_not_raise():
    agent = _make_agent()
    # These exist purely for interface parity with DQN/JEPA agents.
    agent.remember(np.random.rand(S), 0, 1.0, np.random.rand(S), False)
    assert agent.update() is None
    agent.on_env_step()
    agent.increment_episode()
    assert agent.get_stats()['num_params'] == agent.num_params


def test_agent_ask_tell_cycle_advances_generation():
    agent = _make_agent()
    genomes = agent.ask()
    assert len(genomes) == 8
    agent.tell(np.random.rand(8))
    assert agent.generation == 1


def test_agent_save_and_load_roundtrip(tmp_path):
    agent = _make_agent()
    # Drive one generation so there is a meaningful "best genome" to persist.
    agent.tell(np.random.rand(8))

    path = tmp_path / "evo_model.pth"
    agent.save(str(path))
    assert path.exists()

    other = _make_agent(seed=99)  # different init on purpose
    other.load(str(path))

    # Loaded policy must reproduce the saved agent's actions.
    for _ in range(5):
        state = np.random.rand(S).astype(np.float32)
        assert agent.select_action(state) == other.select_action(state)


def test_agent_load_rejects_dim_mismatch(tmp_path):
    agent = _make_agent()
    path = tmp_path / "evo_model.pth"
    agent.save(str(path))

    mismatched = EvolutionAgent(state_dim=S + 1, action_dim=A, hidden_dim=16, seed=0)
    with pytest.raises(ValueError):
        mismatched.load(str(path))
