"""
Configuration and hyperparameters for the RL Racing Game
"""
import numpy as np


class Config:
    # =====================
    # Display Settings
    # =====================
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    TITLE = "PyRacer - RL Racing Game"
    
    # =====================
    # Track Settings
    # =====================
    TRACK_WIDTH = 100  # Width of the track (road width)
    TRACK_MIN_RADIUS = 50  # Minimum turn radius
    TRACK_COMPLEXITY = 8  # Number of control points for generation
    SPLINE_POINTS = 30  # Number of intermediate points between control points (smoother curves)
    NUM_CHECKPOINTS = 4  # Number of checkpoints per lap
    
    # =====================
    # Car Settings
    # =====================
    CAR_WIDTH = 50
    CAR_HEIGHT = 30
    CAR_MAX_SPEED = 200.0  # Maximum speed (pixels per second)
    CAR_ACCELERATION = 400.0  # Acceleration rate (pixels per second^2)
    CAR_BRAKING = 600.0  # Braking rate
    CAR_FRICTION = 50.0  # Friction coefficient
    CAR_STEERING_SPEED = 120.0  # Degrees per second
    CAR_MAX_STEERING_ANGLE = 30  # Maximum steering angle in degrees
    CAR_STEERING_RETURN_SPEED = 120.0  # Degrees per second for steering return to center
    
    # =====================
    # Physics Settings
    # =====================
    DT = 1.0 / 60.0  # Time delta (assuming 60 FPS)
    ACCELERATION_CURVE = "s_curve"  # Options: "linear", "s_curve", "log_curve"
    
    # =====================
    # Sensor Settings
    # =====================
    NUM_SENSORS = 7  # Number of raycast sensors
    SENSOR_MAX_DISTANCE = 200  # Maximum sensor range
    SENSOR_ANGLES = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])  # Sensor angles relative to car
    
    # =====================
    # RL Settings - DQN
    # =====================
    STATE_DIM = NUM_SENSORS + 4  # Sensors + speed + sin(angle) + cos(angle) + progress
    STATE_VERSION = 2  # Bump when state vector layout changes
    ACTION_DIM = 5  # Number of discrete actions
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    GAMMA = 0.99  # Discount factor
    EPSILON_START = 1.0  # Initial exploration rate
    EPSILON_MIN = 0.01  # Minimum exploration rate
    EPSILON_DECAY = 0.995  # Exploration decay rate
    MEMORY_SIZE = 10000  # Replay buffer size
    BATCH_SIZE = 64
    LEARNING_STARTS = 1000  # Environment steps to collect before training
    TARGET_UPDATE_MODE = "polyak"  # Options: "polyak", "hard"
    TARGET_UPDATE_FREQ = 1000  # Train steps between hard target updates
    POLYAK_TAU = 0.005  # Soft target update rate when TARGET_UPDATE_MODE == "polyak"
    PRIORITIZED_REPLAY_ALPHA = 0.6
    PRIORITIZED_REPLAY_BETA = 0.4
    
    # =====================
    # Training Settings
    # =====================
    NUM_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 2000
    TRAIN_START_EPISODE = 100  # Deprecated: use LEARNING_STARTS for env-step warm-up
    SAVE_FREQ = 50  # Save model every N episodes
    LOG_FREQ = 10  # Log training every N episodes
    DEFAULT_SEED = None  # Set to an int for reproducible runs by default
    
    # =====================
    # Reward Settings
    # =====================
    REWARD_LAP_COMPLETE = 200.0
    REWARD_CHECKPOINT = 5.0
    REWARD_COLLISION = -50.0
    REWARD_TIME_PENALTY = -0.01  # Per step penalty to encourage speed
    REWARD_PROGRESS = 25.0  # Reward per fractional progress made
    
    # =====================
    # Action Mapping
    # =====================
    ACTIONS = {
        0: {'throttle': 1.0, 'steering': 0.0},      # Accelerate straight
        1: {'throttle': 0.8, 'steering': -0.8},    # Accelerate + turn left
        2: {'throttle': 0.8, 'steering': 0.8},     # Accelerate + turn right
        3: {'throttle': 0.3, 'steering': 0.0},     # Coast straight
        4: {'throttle': -1.0, 'steering': 0.0},    # Brake hard
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
