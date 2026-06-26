# T18 — `TRAIN_START_EPISODE` is documented but never honoured 🟠

## Problem

`utils/config.py` defines:

```python
TRAIN_START_EPISODE = 100  # Start training after this many episodes
```

…and `AGENTS.md` mentions it. But `train.py` and `DQNAgent.update`
never check it. `update()` runs every step from episode 1, so the
agent starts training as soon as it has `BATCH_SIZE` transitions —
typically within ~1 second of starting.

A pure-exploration warm-up is standard practice and helps avoid
overfitting to noisy early data.

## Why it matters

- The agent learns from a tiny replay buffer with low coverage.
- The documented behaviour differs from actual behaviour.

## Fix

Gate `update()` behind an env-step threshold:

```python
class DQNAgent:
    def __init__(self, ..., learning_starts: int = 1000):
        ...
        self.learning_starts = learning_starts

    def update(self):
        if self.env_steps < self.learning_starts:
            return None
        ...
```

Either keep the per-episode threshold from config, or — better — switch
to a per-env-step threshold (`learning_starts=1000` is what
rl-baselines3-zoo uses). Plumb it through `train.py`:

```python
parser.add_argument('--learning-starts', type=int, default=1000)
```

## Acceptance criteria

- With `--learning-starts 5000`, the CSV shows `loss=0` for the first
  few thousand env steps, then loss > 0 once training begins.
- Removing the flag (default) gives behaviour equivalent to current
  except after the warm-up.

## Files touched

- `rl/agent.py`
- `utils/config.py` (rename / document)
- `train.py`

## Depends on

T17 (uses `env_steps`).
