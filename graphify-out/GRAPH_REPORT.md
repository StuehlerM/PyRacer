# Graph Report - .  (2026-07-01)

## Corpus Check
- Corpus is ~36,503 words - fits in a single context window. You may not need a graph.

## Summary
- 638 nodes · 919 edges · 56 communities (39 shown, 17 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]

## God Nodes (most connected - your core abstractions)
1. `Track` - 40 edges
2. `DQNAgent` - 28 edges
3. `EvolutionAgent` - 26 edges
4. `JEPAAgent` - 24 edges
5. `RacingEnv` - 22 edges
6. `Game` - 20 edges
7. `RandomAgent` - 19 edges
8. `TrainingLogger` - 17 edges
9. `Population` - 16 edges
10. `train_agent()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `env()` --calls--> `RacingEnv`  [INFERRED]
  tests/test_environment.py → game/environment.py
- `test_random_agent_actions_in_range()` --calls--> `RandomAgent`  [INFERRED]
  tests/test_agent.py → rl/agent.py
- `TrainingMetrics` --uses--> `Track`  [INFERRED]
  compare.py → game/track.py
- `TrainingMetrics` --uses--> `RacingEnv`  [INFERRED]
  compare.py → game/environment.py
- `TrainingMetrics` --uses--> `DQNAgent`  [INFERRED]
  compare.py → rl/agent.py

## Communities (56 total, 17 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (23): Represents a 2D racing track with procedural generation.          The track is, Generate smooth Catmull-Rom spline through control points (closed loop)., Smooth the waypoints to create a more natural track (fallback method)., Initialize a new track.                  Args:             width: int - scree, Create smooth inner and outer boundaries from the center waypoints.         Use, Create checkpoints along the track for progress measurement., Get a point in the center of the track at a specific offset., Update the cached bounding box for the track. (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (20): Initialize the racing environment.                  Args:             track:, Game, Execute one game step.                  Args:             action: int or tupl, Main game class that manages the game state and loop.          Supports both h, Initialize the game.                  Args:             track: Track - custom, True if the car left the screen by more than config.OOB_MARGIN pixels., Forward progress distance from current progress to next ordered checkpoint., Get the current state observation for RL.                  Args: (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (33): Config, ConvDQN, Car, compare.py, config, conftest, DQNAgent, DQN (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (14): EvolutionAgent, Select an action with the currently active policy.          Evolution does not, Return the genomes to evaluate this generation (list of vectors)., Load a genome's weights into the policy network for evaluation.          Args:, Report fitnesses and breed the next generation.          Afterwards the best-s, Load the best genome seen so far into the policy network., No-op: evolution scores whole episodes, not individual transitions., No-op: there is no per-step gradient update. Returns None. (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (19): generate_plots(), main(), parse_args(), print_comparison(), Get summary statistics., Train one approach and collect metrics.          Args:         approach: 'dqn, Generate matplotlib comparison plots.          Args:         dqn_metrics: Tra, Save comparison results to CSV and JSON. (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (16): Population, Tournament selection: sample K genomes, return a copy of the fittest., Uniform crossover: each gene is taken from parent A or B by a coin flip., Add zero-mean Gaussian noise scaled by the current mutation std.          This, Return a snapshot of population statistics for logging.          Returns:, A fixed-size population of genomes evolved with a genetic algorithm.      Attr, Args:             num_params: Length of each genome.             pop_size: Pop, Return the genomes that should be evaluated this generation.          Returns: (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (18): compare_agents(), main(), parse_args(), Get summary statistics., Print summary of results., Test an agent on the environment.          Args:         agent: DQNAgent or R, Compare random agent vs trained agent.          Args:         args: command l, Parse command line arguments. (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (23): Car, DQN, DQNAgent, DuelingDQN, EvolutionAgent, Game, PolicyNetwork, Population (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (15): Experience replay buffer that stores transitions for RL training.          All, Initialize the replay buffer.                  Args:             capacity: in, Add a transition to the buffer.                  Args:             state: num, Randomly sample a batch of transitions from the buffer.                  Args:, Return the number of transitions in the buffer., Clear all transitions from the buffer., Get all transitions in the buffer.                  Returns:             list, ReplayBuffer (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (12): RandomAgent, Save the agent's state to a file.                  Args:             path: st, Load the agent's state from a file.                  Args:             path:, Random agent for baseline comparison.     Selects actions randomly without lear, Initialize the random agent.                  Args:             action_dim: i, Select a random action.                  Args:             state: Optional[np, No memory for random agent., No learning for random agent. (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (11): DQNAgent, Select the action with the highest Q-value (no exploration)., Deep Q-Network Agent with experience replay and target network.          Imple, Select an action using epsilon-greedy policy.                  Args:, Update the policy network using a batch from the replay buffer., Advance env-step counter and decay epsilon after replay warm-up., Update target network weights from policy network.                  Uses Polya, Store a transition in the replay buffer.                  Args:             s (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (16): PolicyNetwork, Feed-forward policy mapping a state to discrete action logits.      The networ, Args:             state_dim: Dimension of input state.             action_dim:, Map a batch of states to action logits.          Args:             x: (batch_, _make_agent(), Neuroevolution: policy network, GA population, and the EvolutionAgent.  These, test_act_returns_valid_action(), test_agent_ask_tell_cycle_advances_generation() (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (12): Car, Represents a 2D car with physics for the racing game.          The car has pos, Get the car's state as a dictionary.                  Returns:             di, Initialize a new car.                  Args:             x: float - initial x, Check if the car is colliding with the track.                  Args:, Reset the car to a new position and angle.                  Args:, Draw the car on a pygame screen.                  Args:             screen: p, Draw the car's sensors (for debugging).                  Args:             sc (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (10): ConvDQN, Initialize the Dueling DQN.                  Args:             input_dim: int, Select an action using epsilon-greedy policy.                  Args:, Convolutional DQN for image-based states.          This would be used if we re, Initialize the ConvDQN.                  Args:             input_channels: in, Initialize weights with Xavier uniform initialization., Forward pass through the network.                  Args:             x: torch, Select an action using epsilon-greedy policy for image-based state. (+2 more)

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (11): Logger for training progress., Initialize the logger.                  Args:             log_dir: str - dire, Log an episode to CSV., Save training configuration., Print training summary for an episode., Print final training summary., Train the RL agent.          Args:         args: argparse.Namespace - command, Train a policy with neuroevolution (a genetic algorithm).      Unlike DQN/JEPA (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (9): DQN Agent for reinforcement learning. Implements the Deep Q-Network algorithm w, Experience replay buffers for DQN training., DuelingDQN, Neural network models for reinforcement learning., Dueling Deep Q-Network.          Separates value and advantage streams to impr, Initialize weights with Xavier uniform initialization., Forward pass through the network.                  Args:             x: torch, Test all models to ensure they work correctly. (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (13): circle_polygon_collision(), get_circle_polygon_collision_point(), line_segment_intersection(), point_in_polygon(), point_to_line_distance(), Physics utilities for the racing game including collision detection and line in, Check if a circle collides with a polygon.          Args:         center: tup, Get the collision point and normal for a circle colliding with a polygon. (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (8): JEPAAgent, Store transition for world model training.                  Note: reward and d, Called after each environment step (for interface compatibility)., Called at end of each episode., Get training statistics.                  Returns:             dict with trai, Save JEPA model.                  Args:             path: File path to save t, Load JEPA model.                  Args:             path: File path to load f, JEPA Agent with self-supervised world model and CEM planning.          Provide

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (9): JEPA Agent: Self-supervised world model with energy-based planning.  This is t, Train the world model (encoder + predictor).                  JEPA training ob, _update_goals(), Memory buffers for JEPA training.  Key difference from RL replay buffers:, JEPA (Joint Embedding Predictive Architecture) neural network models.  Unlike, Update target encoder parameters via exponential moving average.          targ, VICReg loss: Variance-Invariance-Covariance Regularization.          This is t, update_target_encoder() (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (8): PrioritizedReplayBuffer, Prioritized experience replay buffer.          Samples transitions based on th, Initialize the prioritized replay buffer.                  Args:, Add a transition with maximum priority (for new transitions)., Update priorities for specific transitions.                  Args:, Sample a batch of transitions using prioritized sampling.                  Arg, Return the number of transitions in the buffer., Clear all transitions from the buffer.

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (5): Evolution Agent: gradient-free, population-based policy search.  This is the t, Policy network for Neuroevolution.  Unlike the DQN/JEPA networks, this network, Total number of scalar weights — the length of the genome vector., set_flat_params(), Genetic algorithm population for Neuroevolution.  This is the heart of the "ev

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (7): GoalBuffer, Stores latent representations of "goal" states for planning.          In RL, t, Args:             capacity: Maximum number of goal embeddings             late, Add a goal latent state.                  Args:             latent_state: (la, Add multiple goal latent states.                  Args:             latent_st, Sample goal states for planning.                  Args:             batch_siz, Get the average goal state (centroid of all goals).                  Using the

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (7): RacingEnv, Render the environment.                  Args:             mode: str - render, Reinforcement learning environment for the racing game.          Provides a st, Set a new track for the environment.                  Args:             track, Reset the environment, possibly changing to a new track.                  Retu, Reset the environment to initial state.                  Returns:, Get environment statistics.                  Returns:             dict: envir

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (11): _fill_memory(), _make_agent(), DQN agent: action selection, the replay-driven update step, and save/load.  Th, test_epsilon_does_not_decay_during_learning_warmup(), test_greedy_action_in_range(), test_on_env_step_decays_epsilon(), test_random_agent_actions_in_range(), test_save_and_load_roundtrip() (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.0
Nodes (10): main(), Verify Python version., Run all verification checks., Verify required packages can be imported., Verify project files exist., Verify game components can be imported (without pygame rendering)., verify_game_components(), verify_imports() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.0
Nodes (6): Buffer for JEPA world model training.          Stores simple (state, action, n, Args:             capacity: Maximum number of transitions to store, Store a transition.                  Args:             state: Current state (, Random sample for world model training.                  Args:             ba, Sample states with high track progress (for goal buffer).                  The, TransitionBuffer

### Community 26 - "Community 26"
Cohesion: 0.0
Nodes (9): Update the car's physics based on input controls.                  Args:, angle_between_vectors(), clamp(), log_curve_acceleration(), Clamp a value between min and max., Apply S-curve acceleration that starts fast and slows as approaching max speed., Apply log-curve acceleration that feels more natural.          Args:, Calculate the angle between two vectors in radians.          Args:         v1 (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.0
Nodes (6): MultiTrackEnv, Environment that cycles through multiple tracks.          Useful for training, Initialize the multi-track environment.                  Args:             nu, Execute one step.                  Args:             action: action to take, Get environment statistics., Execute one environment step.                  Args:             action: int

### Community 28 - "Community 28"
Cohesion: 0.0
Nodes (9): _evaluate_genome(), main(), parse_args(), Seed Python, NumPy, and torch random sources., Parse command line arguments., Score one genome (policy) by letting it drive for a few episodes.      Fitness, Test a trained agent.          Args:         args: argparse.Namespace - comma, set_seed() (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.0
Nodes (4): Car class for the racing game with physics and rendering., RL Environment wrapper for the racing game. Provides a Gym-like interface for r, Main game class that manages the game loop, rendering, and RL interface., Track generation and representation for the racing game. Generates procedural 2

### Community 30 - "Community 30"
Cohesion: 0.0
Nodes (5): Predictor, Args:             latent_dim: Dimension of latent space             action_dim, Predict next latent state.                  Args:             z: (batch_size,, Args:             state_dim: Dimension of input state (default: 11), Predicts the next latent state given current latent state and action.

### Community 32 - "Community 32"
Cohesion: 0.0
Nodes (4): env(), Environment contract: reset/step return the documented (obs, reward, done, info), A full-throttle-straight policy leaves the curved track and crashes,     provin, test_episode_runs_to_termination()

### Community 33 - "Community 33"
Cohesion: 0.0
Nodes (7): Get sensor readings from the car's sensors.                  Uses raycasting t, Cast a ray from start in direction and find the first intersection     with any, Return segment start/end arrays from cached arrays or tuple segments., Cast a ray against many segments using vectorized line intersection.      Args, ray_cast(), ray_cast_batch(), _segments_to_arrays()

### Community 34 - "Community 34"
Cohesion: 0.0
Nodes (3): Neural network models: output shapes, epsilon-greedy bounds, input validation., Dueling Q = V + (A - mean(A)); the advantage stream must be mean-zero     relat, test_dueling_decomposition_is_mean_centered()

### Community 35 - "Community 35"
Cohesion: 0.0
Nodes (4): Encodes raw state (sensor readings) into a latent representation.          Thi, Xavier initialization for stable training start., Encode state into latent representation.                  Args:             s, StateEncoder

### Community 36 - "Community 36"
Cohesion: 0.0
Nodes (4): DQN, Deep Q-Network for the racing game.          A simple feed-forward neural netw, Initialize weights with Xavier uniform initialization., Forward pass through the network.                  Args:             x: torch

### Community 37 - "Community 37"
Cohesion: 0.0
Nodes (3): Colors, Config, Configuration and hyperparameters for the RL Racing Game

### Community 39 - "Community 39"
Cohesion: 0.0
Nodes (3): Initialize JEPA agent.                  Args:             state_dim: Dimensio, create_target_encoder(), Create a target encoder as a frozen copy of the online encoder.          The t

### Community 40 - "Community 40"
Cohesion: 0.0
Nodes (3): _deterministic(), Shared pytest configuration.  Runs the whole suite headless so it works on CI, Seed every RNG before each test so failures are reproducible.

## Knowledge Gaps
- **251 isolated node(s):** `Parse command line arguments.`, `Set all random seeds for reproducibility.`, `Collects metrics during training for comparison.`, `Log metrics for one episode.`, `Compute rolling mean for smoother plots.` (+246 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.