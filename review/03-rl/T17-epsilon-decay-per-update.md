# T17 — Epsilon decays per `update()`, not per env step 🟠

## Problem

`DQNAgent.update`:

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
self.steps += 1
```

Decay happens **only inside `update()`**, which itself is short-circuited
when `len(memory) < batch_size` (the first ~64 transitions). So:

- The first 64 env steps see no decay (fine).
- After memory fills, `update()` runs once per env step, so decay
  approximates per-step decay (mostly fine).
- **But** if `update()` is ever skipped — e.g. T18 introduces a warm-up
  episode where you only collect data — epsilon stays pinned at 1.0
  for a long time and the run looks broken.

Also: `self.steps` is the "updates" counter, not the env-step counter,
yet `train.py` reports it as "Steps:" in the HUD, misleading anyone
interpreting the logs.

## Why it matters

- Epsilon schedule is coupled to a different variable than documented.
- The "Steps" column in CSVs mixes two concepts.

## Fix

Split into two counters and decay on env steps:

```python
def update(self):
    ...
    self.train_steps += 1
    if self.train_steps % self.target_update_freq == 0:
        self.update_target()
    return loss.item()

def on_env_step(self):
    """Call once per env step from the trainer."""
    self.env_steps += 1
    self.epsilon = max(self.epsilon_min,
                       self.epsilon * self.epsilon_decay)
```

Trainer loop:

```python
for step in episode:
    action = agent.select_action(state)
    ...
    agent.remember(...)
    agent.update()       # may no-op
    agent.on_env_step()  # always decays
```

## Acceptance criteria

- With `--episodes 1 --batch-size 999999` (so `update()` never runs),
  epsilon still decays over the episode.
- `train_steps` and `env_steps` are reported separately in
  `get_stats()`.

## Files touched

- `rl/agent.py`
- `train.py`

## Depends on

— (none)
