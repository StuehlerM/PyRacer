# PyRacer - 2D Reinforcement Learning Racing Game

A complete 2D racing game where an AI agent learns to drive on procedurally generated tracks using PyTorch reinforcement learning. The project demonstrates a full RL pipeline from environment design to agent training and evaluation.

---

## Features

- **Procedural Track Generation**: Random 2D tracks with Catmull-Rom splines for smooth, natural curves
- **Physics-based Car**: Realistic car physics with acceleration, braking, steering, and friction
- **RL Agent**: Double DQN and Dueling DQN implementations with uniform or prioritized experience replay
- **Sensor-based State**: 7 raycast sensors for environment perception (mimics real car sensors)
- **Complete Training Pipeline**: Training, testing, logging, and model checkpointing
- **Multi-track Training**: Support for training on multiple tracks for better generalization
- **Human vs AI Modes**: Play manually or watch the AI learn

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone or download the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Play the Game (Human Mode)

```bash
python main.py
```

**Controls:**
- `UP`: Accelerate
- `DOWN`: Brake/Reverse  
- `LEFT`: Turn left
- `RIGHT`: Turn right
- `R`: Reset car position
- `S` (hold): Show sensor rays
- `L`: Toggle Learning/Human mode
- `SPACE`: Pause
- `ESC`: Quit

### Train the RL Agent

```bash
# Basic training (headless, fastest)
python train.py --episodes 1000

# Train with visualization (slower but you can see the agent learn)
python train.py --episodes 500 --render

# Use Dueling DQN for better performance
python train.py --episodes 1000 --dueling

# Train on multiple tracks for generalization
python train.py --episodes 1000 --multi-track --num-tracks 5

# Continue training from a saved model
python train.py --episodes 500 --load saved_models/best_model.pth
```

### Test a Trained Agent

```bash
# Simple test with visualization
python test.py --model saved_models/best_model.pth --episodes 10 --render

# Compare random vs trained agent
python test.py --model saved_models/best_model.pth --compare --episodes 20

# Test on multiple tracks
python test.py --model saved_models/best_model.pth --multi-track --num-tracks 5 --episodes 5
```

---

## Architecture Overview

The PyRacer project follows a modular architecture with clear separation of concerns:

### 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PyRacer                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐    ┌─────────────┐    │
│  │   game/     │      │    rl/      │    │   utils/    │    │
│  │  Game Loop  │◄────►│ RL Agent    │    │ Config/     │    │
│  │  & Physics  │      │ & Models    │    │ Helpers     │    │
│  └─────────────┘      └─────────────┘    └─────────────┘    │
│          ▲                  ▲  ▼                  ▲         │
│          │                  │  │                  │         │
│  ┌───────┴──────┐  ┌────────┴────────────►┌───────┴───────┐ │
│  │  main.py     │  │                      │ test.py       │ │
│  │  (Human)     │  │                      │ (Evaluation)  │ │
│  └──────────────┘  │                      └───────────────┘ │
│                    │                                        │
│              ┌─────┴─────┐                                  │
│              │ train.py  │                                  │
│              │ (Training)│                                  │
│              └───────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 📁 Project Structure

```
Pyracer/
├── game/
│   ├── __init__.py        # Package exports
│   ├── car.py            # Car physics, sensors, and rendering
│   ├── track.py          # Procedural track generation and collision
│   ├── game.py           # Main game loop, RL interface, reward system
│   └── physics.py        # Physics utilities (raycasting, collision, math)
├── rl/
│   ├── __init__.py        # Package exports
│   ├── agent.py          # DQN Agent with Double DQN and Dueling DQN
│   ├── model.py          # Neural network architectures (DQN, Dueling DQN, ConvDQN)
│   ├── memory.py         # Experience replay buffer
│   └── environment.py    # Gym-like RL environment wrapper
├── utils/
│   ├── __init__.py
│   └── config.py         # All configuration and hyperparameters
├── main.py               # Human play entry point
├── train.py              # Training entry point
├── test.py               # Testing and evaluation entry point
├── requirements.txt      # Dependencies
├── README.md             # This file
└── AGENTS.md             # Instructions for AI agents
```

---

## Detailed Component Breakdown

### 🎮 Game Engine (`game/` package)

#### Track Generation System
- **Algorithm**: Catmull-Rom splines through randomized control points
- **Features**: 
  - Smooth, natural curves with configurable complexity
  - Inner and outer boundaries for collision detection
  - Checkpoints for progress measurement
  - Start/finish line
  - Spatial partitioning grid for performance optimization
- **Optimizations**: Caching, bounding boxes, nearby segment detection

#### Car Physics System
- **State**: Position, velocity, angle, speed, steering angle
- **Physics**: 
  - Acceleration with configurable curves (linear, S-curve, log-curve)
  - Braking and reverse
  - Friction/drag modeling
  - Steering with automatic return-to-center
- **Sensors**: 7 raycast sensors at angles [-90°, -60°, -30°, 0°, 30°, 60°, 90°]
  - Normalized readings (0-1) based on distance to track boundaries
  - Spatial partitioning for fast raycasting

#### Game Loop
- **Modes**: Human play, RL training, RL testing
- **Optimization**: Cached calculations (collision, progress, sensors)
- **Toggle**: Learning mode can be toggled to skip expensive RL calculations during human play

### 🤖 Reinforcement Learning (`rl/` package)

#### Neural Network Models (`model.py`)
- **DQN**: Standard Deep Q-Network (3 hidden layers, ReLU activation)
- **Dueling DQN**: Separates value and advantage streams for better learning
- **ConvDQN**: Convolutional version for potential image-based states

#### DQN Agent (`agent.py`)
- **Algorithm**: Double DQN with improvements
  - Experience replay buffer (10,000 transitions)
  - Target network (Polyak-updated every training step by default)
  - Epsilon-greedy exploration (starts at 1.0, decays to 0.01)
  - Batch training (64 samples per update)
  - Gradient clipping for stability
  - Huber loss for robust training
- **Features**:
  - Model save/load
  - Training statistics tracking
  - RandomAgent for baseline comparison

#### Environment Wrapper (`environment.py`)
- **RacingEnv**: Single track environment with Gym-like interface
- **MultiTrackEnv**: Cycles through multiple tracks for generalization
- **Features**:
  - Already-normalized sensor, speed, heading, and progress state
  - Statistics tracking (lap times, rewards)
  - Rendering support

---

## How It Works

### 🎯 Game Environment

#### Track Generation (10ms per track)
1. Generate control points in a circular pattern with random offsets
2. Create smooth Catmull-Rom spline waypoints through control points
3. Calculate inner and outer boundaries using normals at each waypoint
4. Create collision segments from boundary points
5. Place checkpoints at regular intervals along the track
6. Build spatial grid for fast collision and raycast queries

#### Car Physics (per frame)
1. Apply steering input and return-to-center
2. Apply throttle/brake with selected acceleration curve
3. Apply friction based on current speed
4. Clamp speed to maximum
5. Calculate velocity direction based on angle and steering
6. Update position
7. Keep heading wrapped to [-pi, pi] for stable state encoding

#### Sensor System (per frame, RL mode only)
1. For each of 7 sensors at different angles:
   - Calculate absolute sensor direction (car angle + sensor angle)
   - Use optimized raycast with spatial partitioning
   - Return normalized distance (0-1) based on max range

#### Reward System
- **Lap Complete**: +200 points (+50% bonus for new best lap)
- **Checkpoint**: +5 points
- **Progress**: +25 points per fractional track progress
- **Collision**: -50 points (episode ends)
- **Time Penalty**: -0.01 per step (encourages speed)

### 🧠 Reinforcement Learning

#### State Representation (11 dimensions)
```
State = [
    sensor_0,   # -90° left
    sensor_1,   # -60°
    sensor_2,   # -30°
    sensor_3,   # 0° front
    sensor_4,   # 30°
    sensor_5,   # 60°
    sensor_6,   # 90° right
    speed,       # Normalized car speed (-1 to 1)
    sin_angle,   # Sine of wrapped car heading
    cos_angle,   # Cosine of wrapped car heading
    progress     # Progress along track (0-1)
]
```

#### Action Space (5 discrete actions)
| Action | Throttle | Steering | Description |
|--------|----------|----------|-------------|
| 0      | 1.0      | 0.0      | Full throttle straight |
| 1      | 0.8      | -0.8     | Accelerate + turn left |
| 2      | 0.8      | 0.8      | Accelerate + turn right |
| 3      | 0.3      | 0.0      | Coast straight |
| 4      | -1.0     | 0.0      | Brake hard |

#### Training Loop
```
1. Reset environment → Get initial state
2. Select action using epsilon-greedy policy:
   - With probability ε: random action (exploration)
   - With probability 1-ε: best action from policy network (exploitation)
3. Execute action in environment
4. Observe next state and reward
5. Store transition in replay buffer
6. Sample batch from replay buffer
7. Update policy network using Double DQN:
   - Use policy network to select best action for next state
   - Use target network to estimate value of next state
   - Calculate target: reward + γ * Q_target(next_state, best_action)
   - Minimize Huber loss between current Q and target Q
8. Decay exploration rate ε once per environment step
9. Update target network using configured mode (`polyak` by default)
10. Repeat for all episodes
```

#### Double DQN Improvement
Standard DQN uses target network to both select and evaluate the best action:
```
Q_target = reward + γ * max_a Q_target(next_state, a)
```

Double DQN separates selection and evaluation:
```
# Use policy network to select best action
best_action = argmax_a Q_policy(next_state, a)

# Use target network to evaluate that action
Q_target = reward + γ * Q_target(next_state, best_action)
```

This reduces overestimation bias and leads to more stable training.

#### Dueling DQN Improvement
Separates the Q-value into value and advantage components:
```
Q(s,a) = V(s) + A(s,a) - mean(A(s))
```

Where:
- **V(s)**: Value of the state (how good it is to be in this state)
- **A(s,a)**: Advantage of each action (how much better this action is than others)

This helps the agent learn which states are valuable without having to learn the effect of each action for each state.

## Configuration

All hyperparameters and settings are in `utils/config.py`:

### Display Settings
```python
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60
TITLE = "PyRacer - RL Racing Game"
```

### Track Settings
```python
TRACK_WIDTH = 100
TRACK_COMPLEXITY = 8
SPLINE_POINTS = 30
NUM_CHECKPOINTS = 4
```

### Car Settings
```python
CAR_WIDTH = 50
CAR_HEIGHT = 30
CAR_MAX_SPEED = 200.0
CAR_ACCELERATION = 400.0
CAR_BRAKING = 600.0
CAR_FRICTION = 50.0
CAR_STEERING_SPEED = 120.0
CAR_MAX_STEERING_ANGLE = 30
```

### Sensor Settings
```python
NUM_SENSORS = 7
SENSOR_MAX_DISTANCE = 200
SENSOR_ANGLES = [-90, -60, -30, 0, 30, 60, 90]  # degrees
```

### RL Hyperparameters
```python
STATE_DIM = 11  # 7 sensors + speed + sin(angle) + cos(angle) + progress
STATE_VERSION = 2
ACTION_DIM = 5
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
LEARNING_STARTS = 1000
TARGET_UPDATE_MODE = "polyak"
TARGET_UPDATE_FREQ = 1000
POLYAK_TAU = 0.005
```

### Rewards
```python
REWARD_LAP_COMPLETE = 200.0
REWARD_CHECKPOINT = 5.0
REWARD_COLLISION = -50.0
REWARD_TIME_PENALTY = -0.01
REWARD_PROGRESS = 25.0
```

---

## Performance Optimizations

### Spatial Partitioning
- Track is divided into a grid for fast collision and raycast queries
- Only checks nearby segments instead of all track segments
- Reduces collision detection from O(n) to O(1) for typical scenarios

### Caching
- Bounding boxes for tracks
- Waypoint arrays as NumPy arrays
- Segment lengths and total track length
- Sensor nearby segments cache

### Lazy Evaluation
- Progress calculation only when needed
- Sensor readings skipped in human mode
- Cached values passed through the pipeline to avoid recomputation

### Batch Processing
- Experience replay with random sampling
- GPU-accelerated neural network training (via PyTorch)

---

## Training Tips

### Starting Out
1. **Verify Setup**: Run with `--render` and small episode count first
   ```bash
   python train.py --episodes 50 --render
   ```
2. **Monitor Progress**: Watch the agent's behavior in the first few episodes
3. **Check Rewards**: Ensure the agent is receiving appropriate rewards

### Hyperparameter Tuning
If the agent isn't learning:

| Issue | Try This |
|-------|----------|
| Agent explores too much | Reduce `EPSILON_DECAY` (e.g., 0.997) |
| Agent doesn't explore enough | Increase `EPSILON_DECAY` (e.g., 0.993) |
| Learning is unstable | Reduce `LEARNING_RATE` (e.g., 0.0005) |
| Learning is slow | Increase `BATCH_SIZE` (e.g., 128) |
| Agent forgets old knowledge | Reduce `EPSILON_MIN` (e.g., 0.005) |
| Overestimating Q-values | Use Double DQN (enabled by default) |

### Track Difficulty
If the agent struggles:
- Reduce `TRACK_COMPLEXITY` (e.g., 6)
- Increase `TRACK_WIDTH` (e.g., 120)
- Start with single track before multi-track

### Training Strategies
```bash
# Phase 1: Learn basics (100-500 episodes)
python train.py --episodes 200 --track-complexity 6

# Phase 2: Improve skill (500-1000 episodes)
python train.py --episodes 500 --load saved_models/best_model.pth

# Phase 3: Generalize (1000+ episodes)
python train.py --episodes 1000 --multi-track --num-tracks 3 --load saved_models/best_model.pth

# Use Dueling DQN for faster convergence
python train.py --episodes 1000 --dueling
```

---

## Expected Results

### Training Timeline

| Phase | Episodes | Behavior | Lap Time |
|-------|----------|----------|----------|
| Early | 1-100 | Random actions, many collisions | N/A |
| Mid | 100-500 | Starts following track, occasional laps | 30-60s |
| Late | 500-1000 | Consistent laps, optimizing line | 15-30s |
| Expert | 1000+ | Near-optimal racing lines | 10-20s |

### Performance Metrics

On a modern CPU (Intel i7 / Ryzen 7):
- **Headless training**: ~1-2 minutes per 100 episodes
- **With rendering**: ~5-10 minutes per 100 episodes
- **GPU acceleration**: 2-3x faster with CUDA

### Success Criteria
- **Basic**: Agent completes at least one lap within 1000 episodes
- **Good**: Agent completes 90%+ of laps consistently
- **Excellent**: Agent achieves sub-20s lap times on standard tracks

## Command Line Reference

### Training (`train.py`)

```bash
# Required
--episodes          Number of training episodes (default: 10000)

# Training Options
--render            Render training visualization (slower)
--load             Path to load existing model from
--save-dir         Directory to save models (default: saved_models)
--log-dir          Directory to save training logs (default: logs)

# RL Hyperparameters
--batch-size       Batch size for training (default: 64)
--lr               Learning rate (default: 0.001)
--gamma            Discount factor (default: 0.99)
--epsilon          Initial exploration rate (default: 1.0)
--epsilon-min      Minimum exploration rate (default: 0.01)
--epsilon-decay    Exploration decay rate (default: 0.995)

# Track Options
--multi-track      Train on multiple tracks
--num-tracks       Number of tracks for multi-track (default: 3)

# Algorithm Options
--dueling          Use Dueling DQN (default: False)
--test             Test mode (run trained agent without training)
--test-episodes    Number of test episodes (default: 10)
```

### Testing (`test.py`)

```bash
# Required
--model            Path to trained model file

# Options
--episodes        Number of test episodes (default: 10)
--render          Render the test
--compare        Compare random vs trained agent
--multi-track    Test on multiple tracks
--num-tracks     Number of tracks for multi-track (default: 5)
--output         Output directory for results
--record         Record test results to file
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Pygame not found | `pip install pygame` |
| PyTorch not found | `pip install torch` |
| Slow training | Disable rendering with `--no-render` or reduce episode count |
| Agent not learning | Tune hyperparameters (see Training Tips above) |
| Track generation issues | Adjust `TRACK_COMPLEXITY` in config or use `--track-complexity` flag |
| Out of memory | Reduce `MEMORY_SIZE` or `BATCH_SIZE` |
| CUDA errors | Install CUDA toolkit or use CPU-only PyTorch |

### Debug Mode

- **Sensor Visualization**: Hold `S` key during gameplay to see raycast sensors
- **Mode Indicator**: Top-right corner shows LEARNING or HUMAN mode
- **HUD Information**: Speed, laps, lap time, best time, reward, progress

### Verbose Logging

Training progress is logged to:
- CSV file: `logs/training_<timestamp>.csv` - All episode metrics
- JSON file: `logs/config_<timestamp>.json` - Training configuration

---

## Future Enhancements

### RL Algorithm Improvements
- [ ] PPO or SAC for continuous control
- [ ] Prioritized Experience Replay
- [ ] LSTM for temporal dependencies
- [ ] Curiosity-driven exploration (ICM)
- [ ] Distributional RL (C51, QR-DQN)

### State Representation
- [ ] CNN-based state from rendered frame
- [ ] Relative position encoding
- [ ] Temporal stacking (previous N states)

### Game Features
- [ ] Multiple cars (opponents)
- [ ] Different car types with varying physics
- [ ] Obstacles on track
- [ ] Weather effects (rain, fog)
- [ ] Day/night cycle
- [ ] Different track types (oval, technical, street)

### Training Improvements
- [ ] Curriculum learning (easy to hard tracks)
- [ ] TensorBoard integration
- [ ] Episode replay and visualization
- [ ] Automated hyperparameter tuning

### Testing & Evaluation
- [ ] Unit tests for game components
- [ ] RL algorithm validation
- [ ] Benchmarking suite
- [ ] Comparison with other algorithms

## License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute.

---

## Acknowledgments

- **Algorithm Inspiration**: 
  - DQN: [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
  - Double DQN: [Van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)
  - Dueling DQN: [Wang et al., 2016](https://arxiv.org/abs/1511.06581)
- **Track Generation**: Inspired by Freya Holmer's spline-based road techniques
- **Built With**: PyGame for rendering, PyTorch for RL, NumPy for math

---

*Happy Racing! 🏁*

---

*For AI agents working on this project, see [AGENTS.md](AGENTS.md) for detailed instructions.*
