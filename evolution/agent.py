"""
Evolution Agent: gradient-free, population-based policy search.

This is the third learning paradigm in PyRacer, alongside DQN (reward-based RL)
and JEPA (self-supervised world model). It learns by **neuroevolution**:

    Keep a POPULATION of policy networks. Score each one by letting it drive
    (fitness = episode return). Breed the winners (selection + crossover +
    mutation) into the next generation. Repeat.

No gradients. No replay buffer. No reward bootstrapping or TD targets. The only
feedback the algorithm uses is a single scalar fitness per policy per generation.

COMPARISON TO DQN / JEPA:
┌──────────────┬───────────────────┬────────────────────┬──────────────────────┐
│              │ DQN               │ JEPA               │ Evolution            │
├──────────────┼───────────────────┼────────────────────┼──────────────────────┤
│ Signal       │ reward (per step) │ self-supervised    │ episodic fitness     │
│ Optimizer    │ Adam backprop     │ Adam backprop      │ selection + mutation │
│ Memory       │ replay buffer     │ transition buffer  │ population of genomes│
│ Action pick  │ epsilon-greedy Q  │ CEM planning       │ argmax policy forward│
└──────────────┴───────────────────┴────────────────────┴──────────────────────┘

INTERFACE COMPATIBILITY:
    The agent implements the same ``select_action`` / ``save`` / ``load`` surface
    as DQNAgent and JEPAAgent, so it drops straight into test.py and the
    evaluation loops. ``remember`` / ``update`` exist as no-ops (evolution does
    not learn per step). The generational training loop in train.py drives the
    GA through the extra ``ask`` / ``set_active_genome`` / ``tell`` methods.

TRAINING (driven by train.py's train_evolution):
    for generation:
        genomes = agent.ask()                  # population to evaluate
        for genome in genomes:
            agent.set_active_genome(genome)     # load weights into the policy
            fitness = run_episode(env, agent)   # score by driving
        agent.tell(fitnesses)                   # breed the next generation
"""

import numpy as np
import torch

from utils.config import config
from .model import PolicyNetwork
from .population import Population


# Bump if the saved-genome layout changes in an incompatible way.
EVO_CHECKPOINT_VERSION = 1


class EvolutionAgent:
    """
    Neuroevolution agent wrapping a PolicyNetwork and a GA Population.

    Provides the same inference interface as DQNAgent/JEPAAgent
    (``select_action``, ``save``, ``load``) plus GA orchestration hooks
    (``ask``, ``set_active_genome``, ``tell``, ``use_best``).
    """

    def __init__(
        self,
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=config.EVO_HIDDEN_DIM,
        pop_size=config.EVO_POP_SIZE,
        elite_frac=config.EVO_ELITE_FRAC,
        mutation_std=config.EVO_MUTATION_STD,
        mutation_decay=config.EVO_MUTATION_DECAY,
        tournament_size=config.EVO_TOURNAMENT_SIZE,
        crossover_rate=config.EVO_CROSSOVER_RATE,
        init_std=config.EVO_INIT_STD,
        seed=None,
        **kwargs,  # Accept and ignore DQN/JEPA-specific params for interface compat
    ):
        """
        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of discrete actions.
            hidden_dim: Hidden width of each policy network.
            pop_size: Number of policies per generation.
            elite_frac: Fraction of top policies carried over unchanged.
            mutation_std: Initial Gaussian mutation std.
            mutation_decay: Per-generation mutation annealing factor.
            tournament_size: Candidates per selection tournament.
            crossover_rate: Probability of crossover vs cloning.
            init_std: Std of initial random policy weights.
            seed: Optional seed for reproducible evolution.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The single policy network whose weights we swap in and out per genome.
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.num_params = self.policy.num_params()

        # The evolving population of weight vectors.
        self.population = Population(
            num_params=self.num_params,
            pop_size=pop_size,
            elite_frac=elite_frac,
            mutation_std=mutation_std,
            mutation_decay=mutation_decay,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            init_std=init_std,
            seed=seed,
        )

        # Start with the first genome loaded so select_action works immediately.
        self._active_genome = self.population.genomes[0].copy()
        self.policy.set_flat_params(self._active_genome)

        # --- Interface-compatibility attributes (mirrors DQN/JEPA agents) ---
        self.epsilon = 0.0      # No epsilon in evolution; kept for logging compatibility.
        self.memory = []        # No replay buffer; len(self.memory) == 0 for logging.
        self.env_steps = 0
        self.episodes = 0

    # ------------------------------------------------------------------
    # Inference interface (shared with DQNAgent / JEPAAgent)
    # ------------------------------------------------------------------
    def select_action(self, state, explore=True):
        """
        Select an action with the currently active policy.

        Evolution does not explore within a single policy (exploration is across
        the population), so ``explore`` is accepted for interface compatibility
        but ignored — selection is always the greedy argmax.

        Args:
            state: Current state (1-D array).
            explore: Ignored; present for interface parity with DQN/JEPA.

        Returns:
            int: chosen action.
        """
        return self.policy.act(state)

    # ------------------------------------------------------------------
    # GA orchestration (used by the generational training loop)
    # ------------------------------------------------------------------
    def ask(self):
        """Return the genomes to evaluate this generation (list of vectors)."""
        return self.population.ask()

    def set_active_genome(self, genome):
        """
        Load a genome's weights into the policy network for evaluation.

        Args:
            genome: Flat weight vector of length ``num_params``.
        """
        self._active_genome = np.asarray(genome, dtype=np.float32)
        self.policy.set_flat_params(self._active_genome)

    def tell(self, fitnesses):
        """
        Report fitnesses and breed the next generation.

        Afterwards the best-so-far genome is loaded into the policy so that
        immediate evaluation / checkpointing reflects the strongest policy.

        Args:
            fitnesses: One scalar fitness per genome from :meth:`ask`.
        """
        self.population.tell(fitnesses)
        # Keep the policy pointing at the best genome between generations.
        self.policy.set_flat_params(self.population.best_genome)
        self._active_genome = self.population.best_genome.copy()

    def use_best(self):
        """Load the best genome seen so far into the policy network."""
        self.policy.set_flat_params(self.population.best_genome)
        self._active_genome = self.population.best_genome.copy()

    @property
    def generation(self):
        """Generations completed so far."""
        return self.population.generation

    # ------------------------------------------------------------------
    # No-op hooks so generic per-step loops never crash on an EvolutionAgent
    # ------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        """No-op: evolution scores whole episodes, not individual transitions."""
        pass

    def update(self):
        """No-op: there is no per-step gradient update. Returns None."""
        return None

    def on_env_step(self):
        """Count an environment step (for stats parity with other agents)."""
        self.env_steps += 1

    def increment_episode(self):
        """Count a finished episode."""
        self.episodes += 1

    def get_stats(self):
        """Return training statistics, including population GA stats."""
        stats = {
            'env_steps': self.env_steps,
            'episodes': self.episodes,
            'num_params': self.num_params,
        }
        stats.update(self.population.stats())
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """
        Save the best genome plus enough metadata to rebuild the policy.

        Args:
            path: Destination file path.
        """
        torch.save({
            'evo_version': EVO_CHECKPOINT_VERSION,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'num_params': self.num_params,
            'best_genome': self.population.best_genome,
            'best_fitness': self.population.best_fitness,
            'generation': self.population.generation,
        }, path)

    def load(self, path):
        """
        Load a saved best genome into the policy network.

        Args:
            path: Source file path.

        Raises:
            ValueError: If the checkpoint's architecture does not match this agent.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        ckpt_state_dim = checkpoint.get('state_dim', self.state_dim)
        ckpt_action_dim = checkpoint.get('action_dim', self.action_dim)
        if ckpt_state_dim != self.state_dim or ckpt_action_dim != self.action_dim:
            raise ValueError(
                f"Checkpoint policy has state_dim={ckpt_state_dim}, "
                f"action_dim={ckpt_action_dim}, but agent expects "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}"
            )

        # Rebuild the policy if the saved hidden width differs from the current one.
        ckpt_hidden = checkpoint.get('hidden_dim', self.hidden_dim)
        if ckpt_hidden != self.hidden_dim:
            self.hidden_dim = ckpt_hidden
            self.policy = PolicyNetwork(self.state_dim, self.action_dim, ckpt_hidden)
            self.num_params = self.policy.num_params()

        best_genome = np.asarray(checkpoint['best_genome'], dtype=np.float32)
        self.policy.set_flat_params(best_genome)
        self._active_genome = best_genome.copy()
        # Reflect the loaded best into the population so further evolution can resume from it.
        self.population.best_genome = best_genome.copy()
        self.population.best_fitness = float(checkpoint.get('best_fitness', -np.inf))
