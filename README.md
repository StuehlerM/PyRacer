# PyRacer — 2D Reinforcement Learning Racing Game

An educational project where an AI agent learns to drive on procedurally generated
tracks using PyTorch. It implements **two contrasting learning paradigms** so you can
compare them directly:

- **DQN (reinforcement learning)** — learns from a reward signal (Double / Dueling DQN).
- **JEPA (self-supervised world model)** — learns the *dynamics* of the world and
  plans through them with CEM. **No reward function needed.**

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

# Test a saved model (models are timestamped, e.g. best_model_20240101_120000.pth)
python test.py --model saved_models/best_model_<timestamp>.pth --episodes 10 --render

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

### Reward (DQN only)

| Event | Reward |
|-------|--------|
| Lap complete | +200 (×1.5 on a new best lap) |
| Checkpoint | +5 |
| Progress | +25 × fraction of track gained |
| Collision | -50 (ends episode) |
| Per step | -0.01 (encourages speed) |

### DQN vs JEPA

| | DQN | JEPA |
|---|-----|------|
| Learning signal | Reward from environment | Self-supervised prediction |
| What it learns | Q-values (expected return) | World dynamics in latent space |
| Action selection | ε-greedy on Q-values | CEM planning through the world model |
| Goal | Maximize cumulative reward | Reach high-progress latent states |

**DQN** stores transitions in a replay buffer and minimizes the Huber loss between the
predicted Q-value and `reward + γ·Q_target(next, argmax Q_policy(next))` (Double DQN).

**JEPA** encodes states into a latent space, predicts the next latent from
`(z, action)`, and trains against an EMA target encoder with
[VICReg](https://arxiv.org/abs/2105.04906) to prevent collapse. After a warmup, it
selects actions by planning (Cross-Entropy Method) toward "good" latent states it has
seen — no reward signal involved.

---

## Project layout

```
game/    Car physics, procedural track (Catmull-Rom splines), raycasting, reward logic
rl/      DQN/Dueling/Conv models, DQN agent, replay buffers, Gym-like environment
jepa/    Encoder + predictor world model, VICReg loss, CEM planner, transition/goal buffers
utils/   config.py — every hyperparameter lives here
main.py  train.py  test.py  compare.py   verify.py
tests/   pytest suite (run: python -m pytest -q)
```

All hyperparameters (display, physics, sensors, DQN, JEPA, rewards) are centralized in
**`utils/config.py`** — start there to tweak anything.

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
environment's `reset`/`step` contract, and the DQN agent update loop. Tests run
headless (no display required).

---

## Ideas to extend

PPO/SAC, prioritized replay tuning, LSTM/temporal state, curiosity-driven exploration,
CNN state from rendered frames, curriculum learning, opponents and obstacles.

## License & references

Educational use — modify and share freely.

- DQN — [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- Double DQN — [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)
- Dueling DQN — [Wang et al., 2016](https://arxiv.org/abs/1511.06581)
- JEPA / VICReg — [LeCun, 2022](https://openreview.net/forum?id=BZ5a1r-kVsf) · [Bardes et al., 2022](https://arxiv.org/abs/2105.04906)

Built with PyGame (rendering), PyTorch (learning), NumPy (math).
For contributor/agent notes see [AGENTS.md](AGENTS.md).
