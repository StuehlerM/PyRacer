# T01 — Episode reward is double-counted in `Game.step` 🔴

## Problem

In `game/game.py` inside `Game.step()` (RL/learning-mode branch), the
checkpoint and lap rewards are added directly to `self.episode_reward`,
**and then** the same rewards are folded into the local `reward` variable
that is also added to `self.episode_reward` a few lines later:

```python
# Learning-mode branch
if new_checkpoint_idx > self.last_checkpoint_idx:
    ...
    checkpoint_reward = config.REWARD_CHECKPOINT
    self.episode_reward += checkpoint_reward          # (1) added once
    if is_finish:
        ...
        self.episode_reward += lap_reward             # (1) added once
    ...

reward = 0.0
reward += progress_reward
reward += checkpoint_reward                            # (2) added again …
# (lap_reward is not folded in, so the *step* reward is wrong too)
reward += config.REWARD_TIME_PENALTY

self.episode_reward += reward                          # (2) … here
```

So:
- `checkpoint_reward` is counted **twice** in `episode_reward`.
- `lap_reward` is counted **once** in `episode_reward` but **not** in
  the per-step `reward` returned to the agent — the DQN never sees the
  +100/+150 signal.
- `total_reward` has the same problem.

## Why it matters

- The agent's per-step `reward` signal (the only thing the DQN actually
  uses for learning) **does not contain the lap-completion bonus**, even
  though the config and AGENTS.md both say it should.
- `info['episode_reward']` reported to the trainer and CSV is inflated
  by an extra `+10` per checkpoint, so the logs and "best model"
  comparisons are wrong.

## Fix

Compute `reward` first as a pure local sum, then update `episode_reward`
exactly once:

```python
reward = 0.0
reward += progress_reward
if new_checkpoint_idx > self.last_checkpoint_idx:
    reward += config.REWARD_CHECKPOINT
    self.last_checkpoint_idx = new_checkpoint_idx
    if is_finish:
        lap_reward = config.REWARD_LAP_COMPLETE
        if self.lap_time < self.best_lap_time:
            self.best_lap_time = self.lap_time
            lap_reward *= 1.5
        reward += lap_reward
        self.lap_count += 1
        self.lap_time = 0.0
if is_colliding:
    reward += config.REWARD_COLLISION
    done = True
reward += config.REWARD_TIME_PENALTY

self.episode_reward += reward
self.total_reward   += reward
```

Mirror the same single-source-of-truth refactor in the human branch
(where `reward` should simply stay at 0).

## Acceptance criteria

- Sum of per-step `reward` returned from `Game.step` over an episode
  equals `info['episode_reward']` at episode end (assert this in a test).
- Lap-completion bonus appears in the per-step `reward` of the step where
  the finish line is crossed.
- CSV `total_reward` column changes — re-baseline reward charts.

## Files touched

- `game/game.py` — `Game.step`

## Depends on

— (none; do this first)
