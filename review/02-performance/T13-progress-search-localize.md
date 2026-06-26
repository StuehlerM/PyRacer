# T13 — `Track.get_progress` does a full O(N) search every step 🟡

## Problem

`Track.get_progress` scans **every** waypoint each call to find the
closest segment. With `SPLINE_POINTS=30` and `complexity=8` that's
~250 waypoints → 250 `point_to_line_distance` calls per env step,
plus another full scan from the progress reward path.

## Why it matters

- Second-largest CPU cost after raycasting.
- Scales linearly with track complexity — long tracks slow training.

## Fix

The car can only move ~3 px per tick (max speed 200 px/s ÷ 60 fps).
That means the closest segment index changes by at most ±1 per tick.
Cache it:

```python
class Track:
    ...
    def get_progress(self, position, last_idx_hint=None):
        n = self._cached_waypoints_count
        if last_idx_hint is None:
            search_range = range(n)
        else:
            # Search window: last_idx ± 5 with wrap-around
            search_range = [(last_idx_hint + k) % n for k in range(-5, 6)]
        ...
```

And in `Game.step`, pass the previous closest-idx hint:

```python
self._progress_idx_hint = track_idx_used  # returned alongside progress
```

If you don't want to thread the hint through, store it on the Track
instance (single-car assumption is already implicit).

## Acceptance criteria

- Microbenchmark: 10 000 `get_progress` calls in < 100 ms on a
  default track.
- Unit test: result matches the full-scan version on random positions
  (after a warm-up step).

## Files touched

- `game/track.py`
- `game/game.py` (use the hint)

## Depends on

— (none)
