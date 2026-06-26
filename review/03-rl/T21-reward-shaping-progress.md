# T21 — Reward shape is dominated by the +100/-100 wrap-around spikes 🟡

## Problem

After fixing T05, the per-step reward signal is dominated by:

- Progress reward: typical `0.005 × 1.0 = 0.005` per step.
- Time penalty: `-0.05` per step.
- Checkpoint: `+10` (sparse).
- Lap: `+100` to `+150` (very sparse).
- Collision: `-10` (terminates episode).

The time penalty is **10× larger** than the typical positive progress
signal, so the agent is encouraged to *finish the episode fast* (i.e.
crash early) rather than make slow progress.

## Why it matters

- This is a known DQN failure mode — early in training the only
  positive feedback comes from rare checkpoint events that may never
  happen before crashing.

## Fix

Re-balance:

```python
REWARD_TIME_PENALTY = -0.01        # was -0.05
REWARD_PROGRESS     = 5.0          # was 1.0   (per fractional progress)
REWARD_CHECKPOINT   = 5.0          # was 10.0  (now smaller relative to progress)
REWARD_LAP_COMPLETE = 200.0        # was 100.0
REWARD_COLLISION    = -50.0        # was -10.0 (make crashing clearly bad)
```

These are starting suggestions — the key is to verify that a "drives
straight at full throttle" baseline gets positive cumulative reward
on a typical episode (it currently gets negative because time penalty
> progress reward).

Add an explicit unit/integration check in `tests/` that constructs an
"optimal driver" (drives forward at full throttle until first turn)
and asserts mean reward > 0 over 100 steps.

## Why it matters even more after T01/T05

T01 made lap reward visible to the agent (good). T05 made wrap-around
neutral (good). Now the reward scale issue is the *next* thing
limiting training quality.

## Acceptance criteria

- A simple "constant action 0 (full throttle)" policy averages
  positive reward over the first ~50 steps of an episode.
- Plot of episode return for random vs constant-throttle policies
  shows separation.

## Files touched

- `utils/config.py`
- `tests/` (new — see T34)

## Depends on

T01, T05.
