# AGENTS.md - Instructions for AI Agents Working on PyRacer

This document provides guidance for AI agents (like Mistral Vibe) when working with the PyRacer codebase. It explains the architecture, key components, workflows, and best practices for making changes.

---

## 🎯 Quick Reference

### Project Purpose
PyRacer is a **2D racing game with multiple learning approaches** where:
- A car learns to drive on procedurally generated tracks
- Supports **DQN (reward-based RL)** and **JEPA (self-supervised world model)**
- Uses `--approach {dqn,jepa}` flag to switch between approaches
- Supports both **human play** and **AI training** modes

### Key Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `main.py` | Human play entry point | `main()` - runs game loop |
| `train.py` | Training entry point | `train_agent()`, `test_agent()` |
| `test.py` | Evaluation entry point | `compare_agents()` |
| `compare.py` | DQN vs JEPA comparison | `train_approach()`, `generate_plots()` |
| `game/game.py` | Game loop & RL interface | `Game` class, `reset()`, `step()` |
| `game/environment.py` | Gym-style training environment | `RacingEnv`, `MultiTrackEnv` |
| `game/car.py` | Car physics & sensors | `Car` class, `update()`, `get_sensor_readings()` |
| `game/track.py` | Track generation | `Track` class, `_catmull_rom_spline()` |
| `game/physics.py` | Physics utilities | `ray_cast()`, `line_segment_intersection()` |
| `rl/agent.py` | DQN agent | `DQNAgent`, `RandomAgent` |
| `rl/model.py` | Neural networks | `DQN`, `DuelingDQN`, `ConvDQN` |
| `rl/memory.py` | Experience replay | `ReplayBuffer` |
| `jepa/agent.py` | JEPA agent | `JEPAAgent` (world model + CEM planner) |
| `jepa/model.py` | JEPA networks | `StateEncoder`, `Predictor`, `vicreg_loss()` |
| `jepa/memory.py` | JEPA buffers | `TransitionBuffer`, `GoalBuffer` |
| `utils/config.py` | Configuration | `Config` class with all hyperparameters |

---

## 🏗️ Architecture Overview

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN ENTRY POINTS                          │
├─────────────────────────────────────────────────────────────────┤
│  main.py          train.py          test.py       compare.py    │
│      │               │                 │              │         │
│      ▼               ▼                 ▼              ▼         │
└──────────┬───────────┬─────────────────┬──────────────┬───────┘
           │           │                 │              │
           ▼           ▼                 ▼              ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐
│  game/Game      │ │ rl/agent    │ │ game/envi-  │ │ jepa/    │
│                 │ │             │ │ ronment     │ │ agent    │
│ - Game loop    │ │ - DQNAgent  │ │             │ │          │
│ - Rendering    │ │ - RandomAgent│ │ - RacingEnv│ │- JEPAAgent│
│ - RL interface │ │             │ │ - MultiTrack│ │- CEM plan │
└──────────┬──────┘ └──────┬──────┘ └─────────────┘ └────┬─────┘
           │               │                              │
           ▼               ▼                              ▼
┌─────────────────┐ ┌─────────────┐              ┌────────────────┐
│  game/Car       │ │ rl/model.py │              │ jepa/model.py  │
│  game/Track     │ │ - DQN       │              │ - Encoder      │
│  game/Physics   │ │ - DuelingDQN│              │ - Predictor    │
└─────────────────┘ └─────────────┘              │ - VICReg       │
                           │                      └────────────────┘
                           │                              │
                 ┌─────────┴───┐              ┌───────────┴──────┐
                 │ rl/memory.py │              │ jepa/memory.py   │
                 │- ReplayBuffer│              │- TransitionBuffer│
                 └─────────────┘              │- GoalBuffer      │
                                              └──────────────────┘
                              ▲                        ▲
                              │                        │
                    ┌─────────┴────────────────────────┴──┐
                    │           utils/config.py            │
                    │  All hyperparameters (DQN + JEPA)    │
                    └─────────────────────────────────────┘
```

### Data Flow During Training

```
User runs: python train.py --episodes 1000
           │
           ▼
┌─────────────────────────┐
│   train.py:train_agent()│
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   RacingEnv.reset()      │ ←─ Initial state
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   DQNAgent.select_action()│ ←─ ε-greedy: random or from policy_net
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   RacingEnv.step(action) │ ←─ Execute action, get (state, reward, done, info)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   agent.remember()        │ ←─ Store in ReplayBuffer
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   agent.update()          │ ←─ Sample batch, compute loss, backprop
└─────────┬───────────────┘
          │
          ▼ (every 10 steps)
┌─────────────────────────┐
│   agent.update_target()   │ ←─ Copy policy_net → target_net
└─────────────────────────┘
```

### Data Flow During JEPA Training

```
User runs: python train.py --approach jepa --episodes 1000
           │
           ▼
┌──────────────────────────────────────┐
│   Phase 1: World Model Learning       │
│   (warmup_steps random exploration)   │
├──────────────────────────────────────┤
│                                        │
│   state ──► Encoder(state) ──► z_t     │
│                                │       │
│   predictor(z_t, action) ──► z_pred   │
│                                        │
│   next_state ──► TargetEncoder ──► z_target (no grad, EMA)
│                                        │
│   Loss = VICReg(z_pred, z_target)      │
│         = invariance + variance + cov  │
│                                        │
│   Update encoder + predictor (backprop)│
│   Update target encoder (EMA)          │
└──────────────────────────────────────┘
           │
           ▼ (after warmup)
┌──────────────────────────────────────┐
│   Phase 2: CEM Planning               │
├──────────────────────────────────────┤
│                                        │
│   z_current = Encoder(current_state)  │
│   z_goal = GoalBuffer.mean()           │
│                                        │
│   CEM: sample N action sequences       │
│   Simulate each through Predictor      │
│   Score: ||z_final - z_goal||²        │
│   Keep top-K, refine, repeat           │
│                                        │
│   Execute first action of best plan    │
│   Re-plan every step (MPC)             │
└──────────────────────────────────────┘
```

---

## 📋 Workflow Guidelines

### When Adding New Features

1. **Check config first**: `utils/config.py` contains all hyperparameters
   - Add new parameters here if they're user-configurable
   - Keep defaults reasonable

2. **Follow existing patterns**:
   - Classes: Use docstrings with Args/Returns sections
   - Functions: Type hints where possible
   - Imports: Group standard library, third-party, local
   - Naming: `snake_case` for functions/vars, `PascalCase` for classes

3. **Performance matters**:
   - Use caching where possible (see track.py for examples)
   - Pre-calculate expensive operations
   - Use NumPy for numerical operations
   - Only compute sensors in RL mode (skip in human mode)

### When Modifying Game Physics

**Files to check:**
- `game/car.py` - Car movement, steering, acceleration
- `game/physics.py` - Collision detection, raycasting
- `game/track.py` - Track generation, boundaries
- `utils/config.py` - Physics constants

**Key principles:**
- Physics runs at `config.DT = 1/60` second intervals
- All positions are in pixels
- Angles are in radians
- Velocity = direction × speed

### When Modifying RL Components

**Files to check:**
- `rl/agent.py` - DQN algorithm implementation
- `rl/model.py` - Neural network architectures
- `rl/memory.py` - Experience replay buffer
- `game/environment.py` - State/reward wrapping
- `utils/config.py` - RL hyperparameters

**Key principles:**
- State dimension: 11 (7 sensors + speed + sin(angle) + cos(angle) + progress)
- Action dimension: 5 (see ACTIONS in config.py)
- Reward shaping: Encourages progress, penalizes collisions/time
- Double DQN: Separates action selection and evaluation

### When Modifying JEPA Components

**Files to check:**
- `jepa/agent.py` - World model training + CEM planning
- `jepa/model.py` - Encoder, Predictor, VICReg loss
- `jepa/memory.py` - Transition buffer and goal buffer
- `utils/config.py` - JEPA hyperparameters (JEPA_* prefix)

**Key principles:**
- Same state/action dimensions as RL (11-dim state, 5 discrete actions)
- No reward signal used for learning (only for logging/comparison)
- World model trained via self-supervised prediction in latent space
- VICReg prevents representational collapse (all states → same embedding)
- Goals bootstrapped from high-progress states (no human reward design)
- CEM planning: evaluate candidate action sequences through world model
- Agent interface matches DQNAgent (select_action, remember, update, save, load)

**Tuning JEPA:**
- If agent doesn't plan well: increase `JEPA_WARMUP_STEPS` for better world model
- If world model collapses: increase `JEPA_VICREG_LAMBDA` (variance term)
- If planning is slow: reduce `JEPA_CEM_CANDIDATES` or `JEPA_PLANNING_HORIZON`
- If goals are bad: lower `JEPA_GOAL_PROGRESS_THRESHOLD`

---

## 🔍 Key Components Deep Dive

### 1. State Representation

```python
# State vector (11 dimensions)
state = [
    # Sensors (7 values, normalized 0-1)
    sensor_reading_0,   # -90° (left)
    sensor_reading_1,   # -60°
    sensor_reading_2,   # -30°
    sensor_reading_3,   #  0°  (front)
    sensor_reading_4,   #  30°
    sensor_reading_5,   #  60°
    sensor_reading_6,   #  90° (right)
    
    # Car state (4 values)
    normalized_speed,    # speed / max_speed
    sin_angle,          # sin(wrapped heading)
    cos_angle,          # cos(wrapped heading)
    progress            # 0-1 along track
]
```

**Sensor geometry:**
- 7 rays emanating from car center
- Angles relative to car's forward direction
- Max distance: 200 pixels
- Returns normalized distance (0 = hit immediately, 1 = max distance)

### 2. Action Space

```python
# From utils/config.py
ACTIONS = {
    0: {'throttle': 1.0,  'steering': 0.0},   # Accelerate straight
    1: {'throttle': 0.8,  'steering': -0.8},  # Accelerate + turn left
    2: {'throttle': 0.8,  'steering': 0.8},   # Accelerate + turn right
    3: {'throttle': 0.3,  'steering': 0.0},   # Coast straight
    4: {'throttle': -1.0, 'steering': 0.0},   # Brake hard
}
```

**Action mapping:**
- `throttle`: -1.0 (reverse) to 1.0 (full throttle)
- `steering`: -1.0 (left) to 1.0 (right)

### 3. Reward Function

```python
# From game/game.py step() method

# Progress reward (per step)
progress_diff = current_progress - prev_progress
reward += progress_diff * config.REWARD_PROGRESS  # +25 per fractional progress

# Checkpoint reward
if new_checkpoint_reached:
    reward += config.REWARD_CHECKPOINT  # +5

# Lap completion reward
if lap_completed:
    reward += config.REWARD_LAP_COMPLETE  # +200
    if new_best_lap:
        reward *= 1.5  # Bonus for new best

# Collision penalty
if collision:
    reward += config.REWARD_COLLISION  # -50
    done = True

# Time penalty (per step)
reward += config.REWARD_TIME_PENALTY  # -0.01
```

### 4. DQN Algorithm Flow

```
Initialize:
  policy_net = DQN(state_dim=11, action_dim=5, hidden_dim=128)
  target_net = DQN(state_dim=11, action_dim=5, hidden_dim=128)
  target_net.load_state_dict(policy_net.state_dict())
  memory = ReplayBuffer(capacity=10000)
  optimizer = Adam(policy_net.parameters(), lr=0.001)
  epsilon = 1.0

Training step:
  1. state = env.reset()
  2. action = epsilon_greedy(policy_net, state, epsilon)
  3. next_state, reward, done, info = env.step(action)
  4. memory.push(state, action, reward, next_state, done)
  5. state = next_state
  
  6. If len(memory) >= batch_size:
       states, actions, rewards, next_states, dones = memory.sample(batch_size)
       
       # Double DQN
       with torch.no_grad():
         next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
         next_q_values = target_net(next_states).gather(1, next_actions)
       
       target_q = rewards + gamma * next_q_values * (1 - dones)
       current_q = policy_net(states).gather(1, actions)
       
       loss = smooth_l1_loss(current_q, target_q)
       optimizer.zero_grad()
       loss.backward()
       clip_grad_norm_(policy_net.parameters(), 1.0)
       optimizer.step()
  
  7. env_steps += 1
  8. epsilon = max(epsilon_min, epsilon * epsilon_decay)
  
  9. If target_update_mode == "polyak":
       soft-update target_net every train step
     Else if train_steps % target_update_freq == 0:
       hard-copy policy_net to target_net
```

### 5. Track Generation

**Algorithm: Catmull-Rom Spline**

```python
# From game/track.py _catmull_rom_spline()

# Input: control_points = [(x0,y0), (x1,y1), ..., (xn,yn)]
# Output: smooth waypoints through all control points

for each segment (p0, p1, p2, p3):
    for t in [0, 0.1, 0.2, ..., 0.9]:
        x = 0.5 * ((2*p1.x) +
                   (-p0.x + p2.x) * t +
                   (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * t^2 +
                   (-p0.x + 3*p1.x - 3*p2.x + p3.x) * t^3)
        
        y = 0.5 * ((2*p1.y) +
                   (-p0.y + p2.y) * t +
                   (2*p0.y - 5*p1.y + 4*p2.y - p3.y) * t^2 +
                   (-p0.y + 3*p1.y - 3*p2.y + p3.y) * t^3)
        
        waypoints.append((x, y))
```

**Track boundaries:**
- `waypoints`: Center line of track
- `inner_boundary`: Left edge (car's perspective)
- `outer_boundary`: Right edge (car's perspective)
- `segments`: Line segments for collision detection
- `checkpoints`: Regularly spaced along track

---

## 🛠️ Common Tasks

### Task 1: Add a New Action

**Steps:**
1. Update `ACTIONS` dict in `utils/config.py`
2. Update `ACTION_DIM` in `utils/config.py`
3. Update `state_dim` if action affects state representation
4. Test in `train.py` with `--episodes 10 --render`

**Example:** Adding a "coast left" action:
```python
# In utils/config.py
ACTIONS = {
    ...
    5: {'throttle': 0.3, 'steering': -0.5},  # Coast left
}
ACTION_DIM = 6  # Was 5, now 6
```

### Task 2: Modify Reward Function

**Steps:**
1. Locate reward calculation in `game/game.py` step() method
2. Add/modify reward components
3. Update corresponding config values
4. Test that agent still learns effectively

**Example:** Adding speed-based reward:
```python
# In game/game.py step()
# Add after other rewards
speed_reward = state[7] * 0.1  # 7th element is normalized speed
reward += speed_reward

# In utils/config.py
REWARD_SPEED_BONUS = 0.1
```

### Task 3: Add a New Sensor

**Steps:**
1. Update `NUM_SENSORS` in `utils/config.py`
2. Update `SENSOR_ANGLES` array
3. Update `STATE_DIM` to reflect new state size
4. Update sensor drawing in `game/car.py`
5. Test sensor visualization with `S` key

**Example:** Adding a rear sensor:
```python
# In utils/config.py
NUM_SENSORS = 8  # Was 7
SENSOR_ANGLES = np.deg2rad([-90, -60, -30, 0, 30, 60, 90, 180])  # Added 180°
STATE_DIM = NUM_SENSORS + 4  # Recalculate (sensors + speed + sin + cos + progress)
```

### Task 4: Change Neural Network Architecture

**Steps:**
1. Modify network in `rl/model.py`
2. Update `HIDDEN_DIM` in `utils/config.py` if needed
3. Retrain agent (old models won't be compatible)

**Example:** Adding a fourth hidden layer:
```python
# In rl/model.py DQN class
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # New layer
        self.fc4 = nn.Linear(hidden_dim, output_dim)  # Renamed
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # New
        return self.fc4(x)  # Changed
```

### Task 5: Add New RL Algorithm (e.g., PPO)

**Steps:**
1. Create new file `rl/ppo_agent.py`
2. Implement PPOAgent class with same interface as DQNAgent:
   - `__init__(state_dim, action_dim, ...)`
   - `select_action(state, explore=True)`
   - `update()` - return loss
   - `remember(state, action, reward, next_state, done)`
   - `save(path)`
   - `load(path)`
3. Add to `rl/__init__.py`
4. Update `train.py` to support PPO via command line flag

**Interface example:**
```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, ...):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # ... setup policy, value networks, optimizer
    
    def select_action(self, state, explore=True):
        # Return continuous action for PPO
        return action
    
    def update(self):
        # Perform PPO update
        # Return loss value
        return loss.item()
    
    def remember(self, state, action, reward, next_state, done):
        # Store transition
        pass
    
    def save(self, path):
        # Save model
        pass
    
    def load(self, path):
        # Load model
        pass
```

### Task 6: Modify JEPA World Model

**Steps:**
1. Update `jepa/model.py` — modify Encoder/Predictor architecture
2. Update `jepa/agent.py` if the interface changes
3. Update JEPA hyperparameters in `utils/config.py`
4. Test: `python train.py --approach jepa --episodes 10`

**Example: Adding a deeper encoder:**
```python
# In jepa/model.py StateEncoder class
self.net = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim // 2),  # New bottleneck
    nn.LayerNorm(hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, latent_dim),
)
```

### Task 7: Tune JEPA Planning

**Steps:**
1. Adjust CEM parameters in `utils/config.py`:
   - `JEPA_PLANNING_HORIZON`: Longer = better plans, slower inference
   - `JEPA_CEM_CANDIDATES`: More = better exploration of action space
   - `JEPA_CEM_ITERATIONS`: More = better refinement
2. Modify goal selection in `jepa/agent.py _update_goals()`
3. Test: `python compare.py --episodes 200`

**Example: More aggressive planning:**
```python
# In utils/config.py
JEPA_PLANNING_HORIZON = 20   # Plan further ahead
JEPA_CEM_CANDIDATES = 128    # More candidates
JEPA_CEM_ELITES = 20         # Keep more elites
JEPA_CEM_ITERATIONS = 5      # More refinement
```

---

## 🔧 Configuration Reference

### Display & Rendering
```python
SCREEN_WIDTH = 1000        # Screen width in pixels
SCREEN_HEIGHT = 800         # Screen height in pixels
FPS = 60                   # Target frames per second
TITLE = "PyRacer - RL Racing Game"
```

### Track Generation
```python
TRACK_WIDTH = 100          # Road width in pixels
TRACK_COMPLEXITY = 8       # Number of control points
SPLINE_POINTS = 30        # Points between control points
NUM_CHECKPOINTS = 4       # Checkpoints per lap
```

### Car Physics
```python
CAR_WIDTH = 50             # Car width in pixels
CAR_HEIGHT = 30            # Car height in pixels
CAR_MAX_SPEED = 200.0      # Pixels per second
CAR_ACCELERATION = 400.0   # Pixels per second squared
CAR_BRAKING = 600.0        # Braking force
CAR_FRICTION = 50.0        # Friction coefficient
CAR_STEERING_SPEED = 120.0 # Degrees per second
CAR_MAX_STEERING_ANGLE = 30 # Maximum wheel turn
ACCELERATION_CURVE = "s_curve"  # "linear", "s_curve", "log_curve"
```

### Sensors
```python
NUM_SENSORS = 7
SENSOR_MAX_DISTANCE = 200  # Pixels
SENSOR_ANGLES = [-90, -60, -30, 0, 30, 60, 90]  # Degrees
```

### RL Hyperparameters
```python
STATE_DIM = 11             # 7 sensors + speed + sin(angle) + cos(angle) + progress
STATE_VERSION = 2
ACTION_DIM = 5
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
GAMMA = 0.99              # Discount factor
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.995     # Decay rate per step
MEMORY_SIZE = 10000       # Replay buffer size
BATCH_SIZE = 64
LEARNING_STARTS = 1000
TARGET_UPDATE_MODE = "polyak"
TARGET_UPDATE_FREQ = 1000  # Train steps between hard target updates
POLYAK_TAU = 0.005
```

### Rewards
```python
REWARD_LAP_COMPLETE = 200.0
REWARD_CHECKPOINT = 5.0
REWARD_COLLISION = -50.0
REWARD_TIME_PENALTY = -0.01  # Per step
REWARD_PROGRESS = 25.0      # Per fractional progress
```

### Training
```python
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 2000
LEARNING_STARTS = 1000  # Start training after N environment steps
TRAIN_START_EPISODE = 100  # Deprecated; kept for compatibility
SAVE_FREQ = 50           # Save model every N episodes
LOG_FREQ = 10            # Log every N episodes
DEFAULT_APPROACH = "dqn" # "dqn" or "jepa"
```

### JEPA Hyperparameters
```python
# World Model
JEPA_LATENT_DIM = 64         # Latent representation dimension
JEPA_HIDDEN_DIM = 128        # Hidden layer size
JEPA_ENCODER_LR = 0.0003     # Encoder learning rate
JEPA_PREDICTOR_LR = 0.0003   # Predictor learning rate
JEPA_EMA_TAU = 0.005         # Target encoder EMA rate

# VICReg (prevents collapse)
JEPA_VICREG_LAMBDA = 25.0    # Variance coefficient
JEPA_VICREG_MU = 25.0        # Invariance coefficient
JEPA_VICREG_NU = 1.0         # Covariance coefficient

# CEM Planning
JEPA_PLANNING_HORIZON = 10   # Steps to plan ahead
JEPA_CEM_CANDIDATES = 64     # Action sequences per iteration
JEPA_CEM_ELITES = 10         # Top-k sequences kept
JEPA_CEM_ITERATIONS = 3      # Refinement iterations

# Training
JEPA_WARMUP_STEPS = 2000     # Random exploration before planning
JEPA_BATCH_SIZE = 128        # World model batch size
JEPA_MEMORY_SIZE = 50000     # Transition buffer capacity
JEPA_GOAL_BUFFER_SIZE = 1000 # Goal embedding capacity
JEPA_GOAL_PROGRESS_THRESHOLD = 0.3  # Min progress for goals
JEPA_TRAIN_FREQ = 4          # Train every N env steps
```

---

## 🐛 Debugging Guide

### Symptom: Agent Not Learning

**Checklist:**
1. Verify rewards are non-zero
   - Add `print(f"Reward: {reward}")` in training loop
   - Expected: Mix of positive and negative rewards

2. Check state values
   - Add `print(f"State: {state}")` before action selection
   - Expected: Sensor values 0-1, speed 0-1, progress 0-1

3. Verify actions are varied
   - Add `print(f"Action: {action}")` after selection
   - Expected: All 5 actions should be tried (especially early training)

4. Check Q-values
   - In `agent.py`, add q-value logging
   - Expected: Q-values should change over time

**Common causes:**
- Learning rate too high/low
- Epsilon not decaying properly
- Rewards not shaping behavior correctly
- State representation issues

### Symptom: Slow Training

**Checklist:**
1. Disable rendering: `--no-render` flag
2. Reduce batch size: `--batch-size 32`
3. Reduce memory size: `MEMORY_SIZE = 5000`
4. Use CPU-only PyTorch if GPU not available
5. Profile with: `python -m cProfile -s time train.py --episodes 10`

**Common bottlenecks:**
- Raycasting (sensor computation)
- Collision detection
- Neural network forward pass
- Experience replay sampling

### Symptom: Collision Detection Issues

**Checklist:**
1. Visualize track boundaries
   - In `track.py draw()`, add debug drawing
2. Check car collision radius
   - In `car.py check_collision()`, radius calculation
3. Verify track segment generation
   - In `track.py _create_boundaries()`

**Debug code:**
```python
# In game/game.py step()
if is_colliding:
    print(f"Collision at {self.car.position}")
    print(f"Radius: {radius}")
```

### Symptom: Track Generation Problems

**Checklist:**
1. Visualize control points
   - Already drawn as red dots in `track.py draw()`
2. Check spline generation
   - In `track.py _catmull_rom_spline()`
3. Verify boundary calculation
   - In `track.py _create_boundaries()`

**Debug code:**
```python
# In track.py __init__()
print(f"Generated {len(self.waypoints)} waypoints")
print(f"Inner boundary: {len(self.inner_boundary)} points")
print(f"Outer boundary: {len(self.outer_boundary)} points")
```

### Symptom: JEPA World Model Not Learning

**Checklist:**
1. Check VICReg loss components
   - `agent._last_loss` dict contains invariance, variance, covariance
   - Expected: All three should decrease over time
   - If variance stays high → encoder is collapsing

2. Verify buffer has enough data
   - `len(agent.memory)` should exceed `JEPA_BATCH_SIZE`
   - `agent.env_steps` should exceed `JEPA_TRAIN_FREQ`

3. Check latent representations
   ```python
   # Debug: check encoder output variance
   import torch
   states_t = torch.as_tensor(agent.memory.states[:100], device=agent.device)
   z = agent.encoder(states_t)
   print(f"Latent variance per dim: {z.var(dim=0).mean().item():.4f}")
   # Should be > 0.1; if near 0, collapse is occurring
   ```

**Common causes:**
- EMA tau too high (target encoder moves too fast → instability)
- VICReg lambda too low (variance not enforced → collapse)
- Batch size too small (poor variance estimates)

### Symptom: JEPA Agent Not Improving After Warmup

**Checklist:**
1. Check goal buffer population
   - `len(agent.goal_buffer)` should be > 0
   - If 0: car never reaches progress threshold during random exploration
   - Fix: lower `JEPA_GOAL_PROGRESS_THRESHOLD`

2. Verify CEM planning
   ```python
   # Debug: check that planning returns varied actions
   actions = [agent.select_action(state, explore=False) for _ in range(20)]
   print(f"Action distribution: {np.bincount(actions, minlength=5)}")
   # Should not be all one action
   ```

3. Check world model accuracy
   ```python
   # Debug: compare predicted vs actual next state
   z = agent.encoder(states_t[:1])
   z_pred = agent.predictor(z, torch.tensor([action], device=agent.device))
   z_actual = agent.target_encoder(next_states_t[:1])
   error = (z_pred - z_actual).pow(2).mean().item()
   print(f"Prediction error: {error:.4f}")  # Should decrease over training
   ```

---

## 📊 Testing & Validation

### Unit Testing Strategy

**For game components:**
```python
# Test track generation
from game.track import Track
track = Track(complexity=8)
assert len(track.waypoints) > 0
assert len(track.inner_boundary) == len(track.outer_boundary)

# Test car physics
from game.car import Car
car = Car(100, 100, 0)
car.update(throttle=1.0, steering=0.0)
assert car.speed > 0
```

**For RL components:**
```python
# Test agent
from rl.agent import DQNAgent
import numpy as np
agent = DQNAgent(state_dim=11, action_dim=5)
state = np.random.rand(11)
action = agent.select_action(state)
assert 0 <= action < 5

# Test model
from rl.model import DQN
import torch
model = DQN(input_dim=11, output_dim=5, hidden_dim=64)
input_tensor = torch.randn(1, 11)
output = model(input_tensor)
assert output.shape == (1, 5)
```

**For JEPA components:**
```python
# Test JEPA agent
from jepa.agent import JEPAAgent
import numpy as np
agent = JEPAAgent(state_dim=11, action_dim=5)
state = np.random.rand(11).astype(np.float32)
action = agent.select_action(state, explore=True)
assert 0 <= action < 5

# Test world model
from jepa.model import StateEncoder, Predictor
import torch
encoder = StateEncoder(state_dim=11, latent_dim=64)
predictor = Predictor(latent_dim=64, action_dim=5)
z = encoder(torch.randn(1, 11))
assert z.shape == (1, 64)
z_next = predictor(z, torch.tensor([2]))
assert z_next.shape == (1, 64)

# Test VICReg loss
from jepa.model import vicreg_loss
loss = vicreg_loss(torch.randn(32, 64), torch.randn(32, 64))
assert 'total' in loss and loss['total'].item() > 0
```

### Integration Testing

**Manual play test:**
```bash
python main.py
# Verify: Car moves, sensors show, track renders, no crashes
```

**DQN training smoke test:**
```bash
python train.py --approach dqn --episodes 10 --render
# Verify: Training starts, agent moves, rewards update
```

**JEPA training smoke test:**
```bash
python train.py --approach jepa --episodes 10
# Verify: World model trains, loss decreases, no crashes
```

**Comparison test:**
```bash
python compare.py --episodes 20 --no-plot
# Verify: Both approaches train, comparison table prints
```

**Model save/load test:**
```bash
python train.py --episodes 50 --save-dir test_save
ls test_save  # Should show saved models
python test.py --model test_save/model_*.pth --episodes 1 --render
```

---

## 📝 Code Style Guide

### Imports
```python
# Standard library
import os
import sys
import time

# Third-party
import numpy as np
import pygame
import torch
import torch.nn as nn

# Local
from utils.config import config
from .physics import ray_cast
```

### Classes
```python
class ClassName:
    """
    Brief description of class purpose.
    
    More detailed explanation if needed.
    
    Attributes:
        attr1: Description of attribute 1
        attr2: Description of attribute 2
    """
    
    def __init__(self, param1, param2=default):
        """
        Initialize the class.
        
        Args:
            param1: Description of param1
            param2: Description of param2 (default: default)
        """
        self.attr1 = param1
        self.attr2 = param2
    
    def method_name(self, param):
        """
        Description of method.
        
        Args:
            param: Description of parameter
            
        Returns:
            Description of return value
        """
        # Implementation
        return result
```

### Functions
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of function.
    
    Args:
        param1: Description
        param2: Description (default: default)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: If something goes wrong
    """
    # Implementation
    return result
```

### Docstrings
- Use Google style docstrings
- Include Args, Returns, Raises sections where applicable
- Be concise but informative
- Update docstrings when changing function behavior

### Comments
- Use inline comments sparingly
- Prefer self-documenting code
- Use comments to explain WHY, not WHAT
- For complex algorithms, add reference to source

**Bad:**
```python
x = x + 1  # Increment x
```

**Good:**
```python
# Increment x to account for 1-based indexing in track segments
x = x + 1
```

---

## 🔄 Git Workflow

### Commit Messages
```
Type: Brief description (50 chars or less)

Optional body with more details.

- Use present tense
- Capitalize first letter
- No period at end

Types:
- feat: New feature
- fix: Bug fix
- refactor: Code refactoring
- docs: Documentation changes
- test: Testing changes
- perf: Performance improvements
- chore: Maintenance tasks
```

**Examples:**
```
feat: Add Dueling DQN implementation

- Add DuelingDQN class to rl/model.py
- Update agent.py to support dueling option
- Update config with dueling flag

fix: Correct sensor angle calculation

- Fix trigonometry in car.py get_sensor_readings()
- Add unit test for sensor angles

refactor: Optimize collision detection

- Add spatial partitioning to track.py
- Cache nearby segments for raycasting
- Reduces collision checks from O(n) to O(1)
```

### Pull Requests
1. Create feature branch: `git checkout -b feat/feature-name`
2. Make commits with descriptive messages
3. Push branch: `git push origin feat/feature-name`
4. Create PR with clear description
5. Include screenshots if UI changes
6. Reference relevant issues

---

## 📚 Learning Resources

### Reinforcement Learning
- [DQN Paper](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Double DQN Paper](https://arxiv.org/abs/1509.06461) - Van Hasselt et al., 2016
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581) - Wang et al., 2016
- [RL Course by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

### JEPA & Self-Supervised Learning
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) - LeCun, 2022 (JEPA concept paper)
- [I-JEPA Paper](https://arxiv.org/abs/2301.08243) - Assran et al., 2023 (Image JEPA)
- [VICReg Paper](https://arxiv.org/abs/2105.04906) - Bardes et al., 2022 (anti-collapse regularization)
- [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber, 2018 (learned world models)
- [CEM for Planning](https://arxiv.org/abs/1805.12114) - Chua et al., 2018 (PETS: CEM with learned models)

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### PyGame
- [PyGame Documentation](https://www.pygame.org/docs/)
- [PyGame Tutorials](https://www.pygame.org/wiki/tutorials)

### Math & Algorithms
- [Catmull-Rom Spline](https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline)
- [Ray Casting](https://en.wikipedia.org/wiki/Ray_casting)
- [Line Segment Intersection](https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect)

---

## 🎓 Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Update STATE_DIM
**Problem:** Adding a sensor but forgetting to update `STATE_DIM`
**Solution:** Always update `STATE_DIM = NUM_SENSORS + 4` in config

### Pitfall 2: Mismatched Action Space
**Problem:** Changing ACTIONS dict but not ACTION_DIM
**Solution:** Keep `ACTION_DIM = len(ACTIONS)`

### Pitfall 3: Not Normalizing States
**Problem:** Neural network struggles with large state values
**Solution:** Normalize all state components (sensors already are 0-1)

### Pitfall 4: Breaking Target Network Sync
**Problem:** Modifying target network directly
**Solution:** Always use `update_target()` to copy policy to target

### Pitfall 5: Memory Leaks in Pygame
**Problem:** Pygame surfaces not being freed
**Solution:** Always call `game.close()` or `pygame.quit()`

### Pitfall 6: Random Seed Issues
**Problem:** Results not reproducible
**Solution:** Set random seeds consistently:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### Pitfall 7: JEPA Representational Collapse
**Problem:** All states map to same latent embedding; world model useless
**Solution:** Increase VICReg variance coefficient (`JEPA_VICREG_LAMBDA`), verify variance loss is decreasing. Check that latent dimensions have variance > 0.

### Pitfall 8: JEPA Agent Not Planning (Always Random)
**Problem:** Agent continues random actions past warmup
**Solution:** Check that goal buffer is populated. Verify `JEPA_GOAL_PROGRESS_THRESHOLD` isn't too high — car needs to reach that progress during random exploration.

### Pitfall 9: Mixing DQN and JEPA Model Files
**Problem:** Loading a DQN .pth file into JEPAAgent or vice versa
**Solution:** JEPA saves encoder/predictor/target_encoder. DQN saves policy_net/target_net. They are not interchangeable. Use `--approach` flag consistently.

### Pitfall 10: Forgetting --approach Flag When Testing
**Problem:** Testing a JEPA model with DQN agent (or vice versa)
**Solution:** Always match `--approach` between train and test:
```bash
python train.py --approach jepa --save-dir models_jepa
python test.py --approach jepa --model models_jepa/best_model_*.pth
```

---

## 🆘 Getting Help

### For AI Agents (Vibe, etc.)
1. Read this file (AGENTS.md) thoroughly
2. Check the architecture diagrams
3. Follow the workflow guidelines
4. Use the debugging guide for issues
5. Refer to existing code patterns

### For Human Developers
1. Check README.md for user documentation
2. Review plan.md for project history
3. Check existing issues/PRs in Git
4. Consult the Acknowledgments section for references

---

*Last updated: June 30, 2026*
*Maintainer: PyRacer Team*
