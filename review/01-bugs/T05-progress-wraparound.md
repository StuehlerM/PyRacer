# T05 — Lap completion ignores progress wrap-around 🟠

## Problem

`Game.step` computes:

```python
progress_diff = current_progress - self.prev_progress
self.prev_progress = current_progress
progress_reward = progress_diff * config.REWARD_PROGRESS
```

`current_progress` is a value in `[0, 1]` returned by
`Track.get_progress()`. When the car crosses the start/finish line it
wraps from ~0.99 → ~0.01, which makes `progress_diff ≈ -0.98` and
yields a `progress_reward ≈ -98`, dwarfing the +100 lap reward (when
the lap reward even reaches the agent — see T01).

## Why it matters

- The single most important event in the episode (completing a lap)
  results in a *net negative* reward signal for the agent.
- DQN learns to **avoid** crossing the line. You can observe this in
  training: episodes plateau at progress ≈ 0.95.

## Fix

Wrap-aware diff:

```python
raw_diff = current_progress - self.prev_progress
# If we apparently went backwards by > 0.5 of the track, treat it as a wrap.
if raw_diff < -0.5:
    raw_diff += 1.0
elif raw_diff > 0.5:
    # Teleport / glitch — treat as zero progress this step
    raw_diff = 0.0
progress_reward = raw_diff * config.REWARD_PROGRESS
self.prev_progress = current_progress
```

A clean alternative: track *unwrapped* progress (a monotonically
increasing float that just keeps adding when you cross 1.0), and only
take `mod 1.0` for display / sensors.

## Acceptance criteria

- Synthetic unit test: stepping the env from `prev_progress=0.99` to
  `current_progress=0.02` gives `progress_reward ≈ +0.03`, not −0.97.
- Training CSV: `total_reward` no longer has the characteristic
  `-100`-spike on lap-completion steps.

## Files touched

- `game/game.py`

## Depends on

T01 (rebase reward changes together).
