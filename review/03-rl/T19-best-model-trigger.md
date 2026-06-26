# T19 — "Best model" is saved every time the bogus lap time improves 🟠

## Problem

`train.py`:

```python
if info.get('lap_completed', False):
    lap_completed = True
    current_lap_time = info.get('lap_time', 0)
    if current_lap_time < best_lap_time:
        best_lap_time = current_lap_time
        logger.save_model(agent, is_best=True)
```

Combined with T02 (`lap_time` is `0` on the finish-line step), the
**first** lap completion sets `best_lap_time = 0`, and then any future
`current_lap_time` is never smaller — so "best" is whichever model
happened to complete the first lap.

After T02 is fixed, this still uses *one lap* as the criterion, which
is high-variance. A better criterion: average episode reward over the
last N evaluation episodes.

## Why it matters

- The "best model" you ship to `test.py --model best_model_*.pth` is
  often the weakest one that managed any lap at all.
- Reviewers comparing models on the CSV pick the wrong winner.

## Fix

1. Fix T02 first (so `lap_time` is real).
2. Track a moving average of reward (or a separate evaluation rollout
   every N episodes) and save "best" on that.

Pattern:

```python
self.eval_history = deque(maxlen=50)
self.eval_history.append(episode_reward)
avg = np.mean(self.eval_history)
if avg > self.best_eval_score:
    self.best_eval_score = avg
    logger.save_model(agent, is_best=True)
```

Or run a tiny periodic eval (5 episodes with `epsilon=0`) every 100
episodes and save on that.

## Acceptance criteria

- "Best" model is reproducibly the strongest one when running
  `test.py --compare`.
- Files in `saved_models/` are rewritten less frequently (only on real
  improvements).

## Files touched

- `train.py`

## Depends on

T02.
