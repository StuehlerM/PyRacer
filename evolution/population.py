"""
Genetic algorithm population for Neuroevolution.

This is the heart of the "evolution" approach and the part that is *most
different* from DQN/JEPA. There is no gradient, no loss, no optimizer. Instead
we keep a **population** of genomes (flat weight vectors for a PolicyNetwork)
and improve them generation by generation:

    1. ASK   -> hand out the current genomes to be scored (one fitness each).
    2. TELL  -> given the fitnesses, build the next generation:
                 a. ELITISM      keep the top-k genomes unchanged (never lose the best).
                 b. SELECTION    pick parents via tournament selection.
                 c. CROSSOVER    mix two parents (uniform, per-gene coin flip).
                 d. MUTATION     add Gaussian noise to the child's genes.
       Then shrink the mutation scale a little (anneal) and repeat.

Compared to gradient methods:
    Gradient descent moves *one* parameter vector along the loss gradient.
    Evolution searches with a *cloud* of parameter vectors and lets the
    fittest reproduce — a black-box optimizer that needs only a scalar score.

All randomness flows through a single ``numpy`` Generator so runs are
reproducible when seeded.
"""

import numpy as np


class Population:
    """
    A fixed-size population of genomes evolved with a genetic algorithm.

    Attributes:
        num_params: Length of each genome (PolicyNetwork parameter count).
        pop_size: Number of genomes per generation.
        generation: Generations completed so far.
        best_genome: Best genome seen across all generations.
        best_fitness: Fitness of ``best_genome``.
    """

    def __init__(self, num_params, pop_size=50, elite_frac=0.2,
                 mutation_std=0.1, mutation_decay=0.999, tournament_size=3,
                 crossover_rate=0.5, init_std=0.5, seed=None):
        """
        Args:
            num_params: Length of each genome.
            pop_size: Population size (genomes per generation).
            elite_frac: Fraction of top genomes carried over unchanged.
            mutation_std: Initial std of Gaussian mutation noise.
            mutation_decay: Multiplier applied to mutation_std each generation.
            tournament_size: Candidates per selection tournament.
            crossover_rate: Probability of crossover vs. cloning one parent.
            init_std: Std of the initial random genomes.
            seed: Optional seed for reproducible evolution.

        Raises:
            ValueError: If sizes are inconsistent or non-positive.
        """
        if num_params <= 0:
            raise ValueError("num_params must be positive")
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")

        self.num_params = int(num_params)
        self.pop_size = int(pop_size)
        self.mutation_std = float(mutation_std)
        self.mutation_decay = float(mutation_decay)
        self.tournament_size = max(2, int(tournament_size))
        self.crossover_rate = float(crossover_rate)

        # At least one elite so the best policy is never lost between generations.
        self.num_elites = max(1, int(round(elite_frac * self.pop_size)))

        self.rng = np.random.default_rng(seed)

        # Initial genomes: small random weights around zero.
        self.genomes = [
            (init_std * self.rng.standard_normal(self.num_params)).astype(np.float32)
            for _ in range(self.pop_size)
        ]

        self.generation = 0
        self.best_genome = self.genomes[0].copy()
        self.best_fitness = -np.inf
        self._last_mean_fitness = -np.inf
        self._last_max_fitness = -np.inf

    def ask(self):
        """
        Return the genomes that should be evaluated this generation.

        Returns:
            list[np.ndarray]: the current population (length ``pop_size``).
        """
        return self.genomes

    def tell(self, fitnesses):
        """
        Advance one generation given a fitness score per genome.

        Args:
            fitnesses: Sequence of ``pop_size`` scalar fitness values, aligned
                with the genomes returned by :meth:`ask` (higher is better).

        Raises:
            ValueError: If the number of fitnesses does not match the population.
        """
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        if fitnesses.shape[0] != self.pop_size:
            raise ValueError(
                f"Expected {self.pop_size} fitnesses, got {fitnesses.shape[0]}"
            )

        # Rank genomes best-first.
        order = np.argsort(fitnesses)[::-1]
        self._last_mean_fitness = float(np.mean(fitnesses))
        self._last_max_fitness = float(fitnesses[order[0]])

        # Track the all-time best (elitism across generations, not just within one).
        if fitnesses[order[0]] > self.best_fitness:
            self.best_fitness = float(fitnesses[order[0]])
            self.best_genome = self.genomes[order[0]].copy()

        # --- Build the next generation ---
        next_genomes = []

        # 1. Elitism: carry the top genomes through untouched.
        for i in range(self.num_elites):
            next_genomes.append(self.genomes[order[i]].copy())

        # 2. Fill the rest with mutated children of selected parents.
        while len(next_genomes) < self.pop_size:
            parent_a = self._tournament_select(fitnesses)
            if self.rng.random() < self.crossover_rate:
                parent_b = self._tournament_select(fitnesses)
                child = self._crossover(parent_a, parent_b)
            else:
                child = parent_a.copy()
            child = self._mutate(child)
            next_genomes.append(child)

        self.genomes = next_genomes
        self.generation += 1

        # Anneal mutation: explore widely early, fine-tune later.
        self.mutation_std *= self.mutation_decay

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------
    def _tournament_select(self, fitnesses):
        """
        Tournament selection: sample K genomes, return a copy of the fittest.

        Larger tournaments increase selection pressure (greedier); smaller ones
        preserve diversity. Returns a copy so callers can mutate freely.
        """
        idx = self.rng.integers(0, self.pop_size, size=self.tournament_size)
        winner = idx[np.argmax(fitnesses[idx])]
        return self.genomes[winner].copy()

    def _crossover(self, parent_a, parent_b):
        """
        Uniform crossover: each gene is taken from parent A or B by a coin flip.

        This recombines useful "building blocks" from two good policies.
        """
        mask = self.rng.random(self.num_params) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child.astype(np.float32)

    def _mutate(self, genome):
        """
        Add zero-mean Gaussian noise scaled by the current mutation std.

        This is the source of novelty — the analogue of a gradient step, but
        undirected (random) rather than following a loss surface.
        """
        noise = self.rng.standard_normal(self.num_params) * self.mutation_std
        return (genome + noise).astype(np.float32)

    # ------------------------------------------------------------------
    def stats(self):
        """
        Return a snapshot of population statistics for logging.

        Returns:
            dict with generation, best/mean/max fitness and mutation std.
        """
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mean_fitness': self._last_mean_fitness,
            'max_fitness': self._last_max_fitness,
            'mutation_std': self.mutation_std,
            'pop_size': self.pop_size,
            'num_elites': self.num_elites,
        }
