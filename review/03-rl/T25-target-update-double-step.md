# T25 — Target net updates every 10 *training* steps, not as documented 🟡

## Problem

Config:
```python
TARGET_UPDATE_FREQ = 10  # Steps between target network updates
```

AGENTS.md:
> Target Network (updated every 10 steps)

`DQNAgent.update`:
```python
self.steps += 1
if self.steps % self.target_update_freq == 0:
    self.update_target()
```

`self.steps` is *training* steps (and per T17 conflates with env
steps). So target net updates run every ~10 training updates, which —
because we also use Polyak averaging with `tau=0.005` (soft update by
default) — actually performs a *soft* update of 0.5% every 10 training
steps. That's an unusually slow target net.

Standard DDQN: either
- **Hard update** every `target_update_freq` steps (e.g. 1000–10 000),
  OR
- **Polyak averaging** every step with `tau=0.005`.

Mixing them is unusual.

## Why it matters

- The agent learns more slowly than it could.
- Tuning `target_update_freq` doesn't have the effect the docs imply.

## Fix

Pick one path, documented in `Config`:

```python
TARGET_UPDATE_MODE = "polyak"     # "polyak" or "hard"
TARGET_UPDATE_FREQ = 1            # used when mode == "hard"
POLYAK_TAU = 0.005
```

In `DQNAgent.update`:

```python
if self.target_update_mode == "polyak":
    self.update_target()  # soft, every step
else:
    if self.train_steps % self.target_update_freq == 0:
        self.update_target(hard_update=True)
```

## Acceptance criteria

- Documented behaviour matches code.
- Choosing `hard` mode with `freq=1000` produces visible jumps in
  Q-values; choosing `polyak` produces smooth transitions.

## Files touched

- `rl/agent.py`
- `utils/config.py`
- AGENTS.md (already mentioned)

## Depends on

T17 (which adds `train_steps`).
