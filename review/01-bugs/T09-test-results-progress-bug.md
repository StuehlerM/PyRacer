# T09 — `_save_results` in `test.py` records nonsense for progress & collisions 🟡

## Problem

```python
# test.py: _save_results
for j, (reward, lap_time, completed, progress, collisions) in enumerate(zip(
    result.episode_rewards, result.lap_times,
    [True] * len(result.lap_times),                 # always True
    [r * 100 for r in result.episode_rewards],       # progress = reward*100 ?!
    [0] * len(result.episode_rewards)                # collisions always 0
)):
```

The "progress" column is the **reward × 100**, the "completed" column
is always `True`, and the "collisions" column is always `0`. Anyone
analysing the CSV is being lied to.

## Why it matters

- Saved test results are wrong and undermine analysis / model
  selection.
- The bug is in the persistence path only, so models trained with
  good signals are unaffected — but evaluators trust this CSV.

## Fix

`TestResult.add_episode` already takes the real values. Store per-episode
records inside `TestResult` (a list of dicts) and write those out, e.g.:

```python
class TestResult:
    def __init__(self):
        ...
        self.records = []   # list of dicts

    def add_episode(self, reward, lap_time, completed_lap, collision_count, progress, steps):
        self.records.append({
            'reward': reward,
            'lap_time': lap_time,
            'completed_lap': completed_lap,
            'collision_count': collision_count,
            'progress': progress,
            'steps': steps,
        })
        ...
```

Then `_save_results` iterates `result.records` and writes each field
correctly.

## Acceptance criteria

- Sample CSV contains varied `Completed`, `Collisions`, and `Progress`
  values that match what was printed to stdout per episode.

## Files touched

- `test.py`

## Depends on

— (none)
