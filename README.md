# PyRacer — 2D Reinforcement Learning Racing Game

An educational project where an AI agent learns to drive on procedurally generated
tracks using PyTorch. It implements **three contrasting learning paradigms** so you can
compare them directly:

- **DQN (reinforcement learning)** — learns from a reward signal (Double / Dueling DQN).
- **JEPA (self-supervised world model)** — learns the *dynamics* of the world and
  plans through them with CEM. **No reward function needed.**
- **Evolution (neuroevolution)** — evolves a *population* of policy networks with a
  genetic algorithm. **No gradients, no backprop** — just score each policy by driving
  and breed the fittest.

The goal is clarity: the code is heavily commented to teach the concepts, not to ship a game.

---

## Install

Requires Python 3.8+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (use: source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

Sanity-check the setup any time with `python verify.py`.

## Quick start

```bash
# Play it yourself (arrow keys; S = show sensors, R = reset, ESC = quit)
python main.py

# Train DQN (headless is fastest)
python train.py --episodes 1000
python train.py --episodes 1000 --dueling        # Dueling DQN
python train.py --episodes 1000 --render         # watch it learn

# Train JEPA (self-supervised, no rewards)
python train.py --approach jepa --episodes 1000

# Train Evolution (neuroevolution, gradient-free)
python train.py --approach evo --generations 100
python train.py --approach evo --generations 100 --pop-size 50 --render   # watch it evolve

# Test a saved model (models are timestamped, e.g. best_model_20240101_120000.pth)
python test.py --model saved_models/best_model_<timestamp>.pth --episodes 10 --render
python test.py --approach evo --model saved_models/best_model_<timestamp>.pth --episodes 10 --render

# Compare DQN vs JEPA side by side (with plots)
python compare.py --episodes 500
```

Run any entry point with `-h` for the full flag list.

---

## How it works

### State (11 values, all normalized)

7 raycast sensors at `[-90°, -60°, -30°, 0°, 30°, 60°, 90°]` + car speed +
`sin(heading)` + `cos(heading)` + track progress.

### Actions (5 discrete)

| # | Throttle | Steering | Meaning |
|---|----------|----------|---------|
| 0 | 1.0  | 0.0  | Full throttle straight |
| 1 | 0.8  | -0.8 | Accelerate + turn left |
| 2 | 0.8  | 0.8  | Accelerate + turn right |
| 3 | 0.3  | 0.0  | Coast straight |
| 4 | -1.0 | 0.0  | Brake hard |

### Reward (DQN + Evolution)

DQN learns from this signal directly; Evolution uses the per-episode **sum** of it as
each policy's fitness. JEPA ignores it entirely.

| Event | Reward |
|-------|--------|
| Lap complete | +200 (×1.5 on a new best lap) |
| Checkpoint | +5 |
| Progress | +25 × fraction of track gained |
| Collision | -50 (ends episode) |
| Per step | -0.01 (encourages speed) |

### DQN vs JEPA vs Evolution

| | DQN | JEPA | Evolution |
|---|-----|------|-----------|
| Learning signal | Reward from environment | Self-supervised prediction | Episodic fitness (return) |
| Optimizer | Adam (backprop) | Adam (backprop) | Selection + mutation (no gradients) |
| What it learns | Q-values (expected return) | World dynamics in latent space | A population of policy weights |
| Memory | Replay buffer | Transition buffer | Population of genomes |
| Action selection | ε-greedy on Q-values | CEM planning through the world model | argmax of the policy network |
| Goal | Maximize cumulative reward | Reach high-progress latent states | Breed policies that drive farthest |

**DQN** stores transitions in a replay buffer and minimizes the Huber loss between the
predicted Q-value and `reward + γ·Q_target(next, argmax Q_policy(next))` (Double DQN).

**JEPA** encodes states into a latent space, predicts the next latent from
`(z, action)`, and trains against an EMA target encoder with
[VICReg](https://arxiv.org/abs/2105.04906) to prevent collapse. After a warmup, it
selects actions by planning (Cross-Entropy Method) toward "good" latent states it has
seen — no reward signal involved.

**Evolution** keeps a *population* of policy networks. Each generation it scores every
policy by driving (fitness = episode return), then builds the next generation with
**elitism** (keep the best), **tournament selection** (pick parents), **uniform
crossover** (mix two parents), and **Gaussian mutation** (perturb the weights). There is
no backprop, no replay buffer, and no per-step update — a black-box optimizer that needs
only a scalar score per policy. It trains by *generations*, not episodes
(`--generations`, `--pop-size`, `--eval-episodes`).

---

## Project layout

```
game/      Car physics, procedural track (Catmull-Rom splines), raycasting, reward logic, Gym-like environment
rl/        DQN/Dueling/Conv models, DQN agent, replay buffers
jepa/      Encoder + predictor world model, VICReg loss, CEM planner, transition/goal buffers
evolution/ Gradient-free policy net, genetic-algorithm population, neuroevolution agent
utils/     config.py — every hyperparameter lives here
main.py  train.py  test.py  compare.py   verify.py
tests/     pytest suite (run: python -m pytest -q)
```

All hyperparameters (display, physics, sensors, DQN, JEPA, evolution, rewards) are
centralized in **`utils/config.py`** — start there to tweak anything.

---

## Tuning tips

If the DQN agent struggles, edit `utils/config.py`:

| Symptom | Try |
|---------|-----|
| Explores too much / too little | Adjust `EPSILON_DECAY` (0.993–0.997) |
| Unstable learning | Lower `LEARNING_RATE` (e.g. 0.0005) |
| Slow learning | Raise `BATCH_SIZE` (e.g. 128) |
| Track too hard | Lower `TRACK_COMPLEXITY` or raise `TRACK_WIDTH` |

Training logs are written to `logs/` (CSV metrics + JSON config per run); models to
`saved_models/`.

---

## Tests

```bash
python -m pytest -q
```

The suite covers config invariants, the replay buffers, model output shapes, the
environment's `reset`/`step` contract, the DQN agent update loop, and the
neuroevolution policy/population/agent. Tests run headless (no display required).

---

## Ideas to extend

PPO/SAC, prioritized replay tuning, LSTM/temporal state, curiosity-driven exploration,
CNN state from rendered frames, curriculum learning, opponents and obstacles. For the
evolution path: CMA-ES, novelty search, or parallel fitness evaluation.

## License & references

Educational use — modify and share freely.

- DQN — [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- Double DQN — [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)
- Dueling DQN — [Wang et al., 2016](https://arxiv.org/abs/1511.06581)
- JEPA / VICReg — [LeCun, 2022](https://openreview.net/forum?id=BZ5a1r-kVsf) · [Bardes et al., 2022](https://arxiv.org/abs/2105.04906)
- Neuroevolution / Evolution Strategies — [Salimans et al., 2017](https://arxiv.org/abs/1703.03864) · [Such et al., 2017](https://arxiv.org/abs/1712.06567)

Built with PyGame (rendering), PyTorch (learning), NumPy (math).
For contributor/agent notes see [AGENTS.md](AGENTS.md).
