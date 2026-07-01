"""
Configuration and hyperparameters for the RL Racing Game
"""
import numpy as np


class Config:
    # =====================
    # Display Settings
    # =====================
    # These shape window size and frame cadence for humans; agent logic reads state, not screen pixels.
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    TITLE = "PyRacer - RL Racing Game"
    # Episode ends if the car leaves the screen by more than this margin (pixels).
    # Wall-segment collision only fires near a boundary, so this catches a car that
    # escapes the track entirely and would otherwise run until MAX_STEPS_PER_EPISODE.
    OOB_MARGIN = 100
    
    # =====================
    # Track Settings
    # =====================
    # Track generation controls task difficulty by changing curve density, width, and checkpoint spacing.
    TRACK_WIDTH = 100  # Width of the track (road width)
    TRACK_MIN_RADIUS = 50  # Minimum turn radius
    TRACK_COMPLEXITY = 8  # Number of control points for generation
    SPLINE_POINTS = 30  # Number of intermediate points between control points (smoother curves)
    NUM_CHECKPOINTS = 4  # Number of checkpoints per lap
    
    # =====================
    # Car Settings
    # =====================
    # Car dynamics define what actions feel like; stable physics matters before RL tuning matters.
    CAR_WIDTH = 50
    CAR_HEIGHT = 30
    CAR_MAX_SPEED = 200.0  # Maximum speed (pixels per second)
    CAR_ACCELERATION = 400.0  # Acceleration rate (pixels per second^2)
    CAR_BRAKING = 600.0  # Braking rate
    CAR_FRICTION = 50.0  # Friction coefficient
    CAR_LATERAL_FRICTION = 8.0  # Side-slip damping (higher = less drift)
    CAR_STEERING_SPEED = 120.0  # Degrees per second
    CAR_MAX_STEERING_ANGLE = 30  # Maximum steering angle in degrees
    CAR_STEERING_RETURN_SPEED = 120.0  # Degrees per second for steering return to center
    CAR_WHEELBASE = 24.0  # Distance between axles in pixels
    
    # =====================
    # Physics Settings
    # =====================
    # Fixed timestep keeps motion and reward accumulation consistent across fast and slow machines.
    DT = 1.0 / 60.0  # Time delta (assuming 60 FPS)
    ACCELERATION_CURVE = "s_curve"  # Options: "linear", "s_curve", "log_curve"
    
    # =====================
    # Sensor Settings
    # =====================
    # Sensors are agent's "eyes": local distance probes instead of full map observations.
    NUM_SENSORS = 7  # Number of raycast sensors
    SENSOR_MAX_DISTANCE = 200  # Maximum sensor range
    SENSOR_ANGLES = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])  # Sensor angles relative to car
    
    # =====================
    # RL Settings - DQN
    # =====================
    # Observation = sensor distances plus compact car state, enough to drive without raw image input.
    STATE_DIM = NUM_SENSORS + 4  # Sensors + speed + sin(angle) + cos(angle) + progress
    STATE_VERSION = 2  # Bump when state vector layout changes
    # Four forward-only actions keep control problem simple and prevent learned policies from reversing.
    ACTION_DIM = 4  # Number of discrete actions
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    # Gamma near 1.0 values future lap progress almost as much as immediate reward.
    GAMMA = 0.99  # Discount factor
    # Epsilon starts high for exploration, then decays after replay warm-up so the
    # policy does not exploit an untrained Q-network before updates begin.
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_MIN = 0.01  # Minimum exploration rate
    EPSILON_DECAY = 0.999995  # Exploration decay rate per post-warm-up env step
    MEMORY_SIZE = 10000  # Replay buffer size
    BATCH_SIZE = 64
    # Warm-up delays learning until replay buffer has varied data instead of first few biased crashes.
    LEARNING_STARTS = 1000  # Environment steps to collect before training
    TARGET_UPDATE_MODE = "polyak"  # Options: "polyak", "hard"
    TARGET_UPDATE_FREQ = 1000  # Train steps between hard target updates
    POLYAK_TAU = 0.005  # Soft target update rate when TARGET_UPDATE_MODE == "polyak"
    PRIORITIZED_REPLAY_ALPHA = 0.6
    PRIORITIZED_REPLAY_BETA = 0.4
    
    # =====================
    # Training Settings
    # =====================
    # These control training budget, checkpoint cadence, and how often progress is surfaced to user.
    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 2000
    TRAIN_START_EPISODE = 100  # Deprecated: use LEARNING_STARTS for env-step warm-up
    SAVE_FREQ = 50  # Save model every N episodes
    LOG_FREQ = 10  # Log training every N episodes
    DEFAULT_SEED = None  # Set to an int for reproducible runs by default
    
    # =====================
    # Reward Settings
    # =====================
    # Each RL step starts neutral; shaping terms then add small hints or penalties.
    REWARD_INITIAL = 0.0
    # Reward scale makes finishing lap worth much more than risky wall hits or tiny per-step shaping.
    REWARD_LAP_COMPLETE = 200.0
    REWARD_CHECKPOINT = 5.0
    REWARD_COLLISION = -50.0
    REWARD_TIME_PENALTY = -0.01  # Per step penalty to encourage speed
    REWARD_CHECKPOINT_APPROACH = 25.0  # Reward per normalized distance closed to next checkpoint
    REWARD_PROGRESS = REWARD_CHECKPOINT_APPROACH  # Backward-compatible alias
    REWARD_FORWARD_SPEED = 0.02  # Max per-step reward for full-speed track-aligned forward motion
    REWARD_WRONG_WAY_MULTIPLIER = 2.0  # Extra penalty multiplier when progress goes backwards
    REWARD_OFF_TRACK = -25.0  # Penalty applied when car center leaves road width
    OFF_TRACK_TERMINATES = True  # End episode immediately when car leaves track
    REWARD_NO_PROGRESS = -60.0  # Terminal penalty when policy stalls instead of racing
    NO_PROGRESS_PATIENCE_STEPS = 240  # ~4 seconds at 60 FPS with no meaningful forward progress
    MIN_PROGRESS_DELTA = 1e-4  # Smallest forward progress that resets no-progress patience
    
    # =====================
    # Approach Selection
    # =====================
    # Default approach picks classic reward-driven DQN unless caller asks for an alternative.
    DEFAULT_APPROACH = "dqn"  # Options: "dqn", "jepa", "evo"
    
    # =====================
    # JEPA Settings (Joint Embedding Predictive Architecture)
    # =====================
    # JEPA is alternative path: learn compact world model first, then plan actions in latent space.
    # World Model
    JEPA_LATENT_DIM = 64         # Dimension of latent representation z
    JEPA_HIDDEN_DIM = 128        # Hidden layer size for encoder/predictor
    JEPA_ENCODER_LR = 0.0003     # Learning rate for encoder + predictor
    JEPA_PREDICTOR_LR = 0.0003   # Learning rate for predictor head
    
    # EMA Target Encoder (provides stable targets, prevents collapse)
    JEPA_EMA_TAU = 0.005         # EMA interpolation rate (slow-moving target)
    
    # VICReg Regularization (Variance-Invariance-Covariance, prevents collapse)
    JEPA_VICREG_LAMBDA = 25.0    # Variance term coefficient
    JEPA_VICREG_MU = 25.0        # Invariance (prediction) term coefficient
    JEPA_VICREG_NU = 1.0         # Covariance term coefficient
    
    # Planning (CEM - Cross-Entropy Method)
    JEPA_PLANNING_HORIZON = 10   # Steps to plan ahead
    JEPA_CEM_CANDIDATES = 64     # Random action sequences to evaluate per iteration
    JEPA_CEM_ELITES = 10         # Top-k sequences kept each iteration
    JEPA_CEM_ITERATIONS = 3      # CEM refinement iterations
    
    # Training
    JEPA_WARMUP_STEPS = 2000     # Random exploration before planning starts
    JEPA_BATCH_SIZE = 128        # Batch size for world model training
    JEPA_MEMORY_SIZE = 50000     # Transition buffer capacity
    JEPA_GOAL_BUFFER_SIZE = 1000       # Store "good" latent states as planning goals
    JEPA_GOAL_PROGRESS_THRESHOLD = 0.3 # Min track progress to count as "good" state
    JEPA_TRAIN_FREQ = 4          # Train world model every N env steps
    
    # =====================
    # Evolution Settings (Neuroevolution / Genetic Algorithm)
    # =====================
    # Evolution is a third, gradient-free path: instead of backprop on a loss, keep a
    # POPULATION of policy networks and let the fittest reproduce. No reward bootstrap,
    # no replay buffer, no target network — just "score each policy, breed the winners".
    EVO_HIDDEN_DIM = 64          # Hidden width of each policy network (kept small: every weight is a gene)
    EVO_POP_SIZE = 50            # Policies evaluated per generation
    EVO_GENERATIONS = 100        # Default number of generations to evolve
    EVO_ELITE_FRAC = 0.2         # Fraction of top policies carried over unchanged (elitism)
    EVO_TOURNAMENT_SIZE = 3      # Candidates per tournament when selecting parents
    EVO_CROSSOVER_RATE = 0.5     # Probability of recombining two parents vs. cloning one
    EVO_MUTATION_STD = 0.1       # Initial std of Gaussian weight mutations
    EVO_MUTATION_DECAY = 0.999   # Per-generation multiplier that anneals mutation std
    EVO_INIT_STD = 0.5           # Std of the initial random policy weights
    EVO_EVAL_EPISODES = 1        # Episodes averaged to score one policy's fitness
    
    # =====================
    # Action Mapping
    # =====================
    # Action set favors common racing choices while keeping learned policies forward-only.
    ACTIONS = {
        0: {'throttle': 1.0, 'steering': 0.0},      # Accelerate straight
        1: {'throttle': 0.8, 'steering': -0.8},    # Accelerate + turn left
        2: {'throttle': 0.8, 'steering': 0.8},     # Accelerate + turn right
        3: {'throttle': 0.3, 'steering': 0.0},     # Coast straight
    }
    
    # =====================
    # Colors
    # =====================
    class Colors:
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128)
        DARK_GRAY = (64, 64, 64)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        YELLOW = (255, 255, 0)
        CYAN = (0, 255, 255)
        ORANGE = (255, 165, 0)
        TRACK_COLOR = (50, 50, 50)  # Dark gray for track
        TRACK_BORDER = (200, 200, 200)  # Light gray for borders
        ROAD_COLOR = (70, 70, 70)
        GRASS_COLOR = (34, 139, 34)
        CAR_COLOR = (0, 0, 255)
        CHECKPOINT_COLOR = (255, 255, 0)
        FINISH_LINE_COLOR = (255, 0, 0)


# Create a global config instance
config = Config()
