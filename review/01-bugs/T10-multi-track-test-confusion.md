# T10 — `MultiTrackEnv` only swaps tracks every 100 episodes 🟡

## Problem

```python
class MultiTrackEnv:
    def reset(self):
        if self.env.episode > 0 and self.env.episode % 100 == 0:
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks
            self.env.set_track(self.tracks[self.current_track_idx])
        return self.env.reset()
```

If you run `python test.py --multi-track --num-tracks 5 --episodes 5`
the wrapper does `5 × 5 = 25` resets, but the track only changes at
episodes 100, 200, … — so **all 25 test episodes happen on track 0**.
The whole point of multi-track testing is lost silently.

There is also no mechanism to evaluate "1 episode per track" for a
fair comparison.

## Why it matters

- Multi-track evaluation produces single-track results without warning.
- Multi-track training rotation feels arbitrary (why 100?) and is
  un-tunable.

## Fix

- Make rotation cadence a constructor argument:
  `MultiTrackEnv(num_tracks, rotation_every=100)`.
- Add an `eval_mode=True` flag that rotates **every reset** for testing.
- Better default for training: `rotation_every=10` (random domain
  sampling encourages generalisation).

```python
def __init__(self, num_tracks=5, render=False, rotation_every=10, eval_mode=False, ...):
    ...
    self.rotation_every = 1 if eval_mode else rotation_every
```

## Acceptance criteria

- `test.py --multi-track --num-tracks 5 --episodes 5` runs each track
  exactly once.
- `train.py --multi-track` (default config) rotates more frequently
  than every 100 episodes.

## Files touched

- `rl/environment.py`
- `test.py`, `train.py` — pass `eval_mode=True` from test path.

## Depends on

— (none)
