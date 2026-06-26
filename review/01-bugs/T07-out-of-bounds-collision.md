# T07 — `Track.check_collision` returns false positives near outer bounding box 🟠

## Problem

`Track.check_collision` first does:

```python
if (x < self._cached_min_x - radius or x > self._cached_max_x + radius or
    y < self._cached_min_y - radius or y > self._cached_max_y + radius):
    return True, {'type': 'out_of_bounds'}
```

`_cached_min_x` etc. are computed over the **boundary points**, not the
track surface. Anything outside this box is collided.

That's fine in principle, but:

1. The car spawns *on* a centre waypoint (T04 unrelated). If the
   spawn waypoint is the outermost point of the loop, `position == max_x`
   and we trip the boundary `>` check on step 1 because `radius > 0`.
2. The reset path uses the *track* start position which is inside the
   loop, so it's safe in practice — but if a future change uses
   `get_random_start_position`, that picks any waypoint and triggers
   the check.

More importantly, the function is named "check collision" and is
expected to mean "you crashed". `out_of_bounds` is treated identically
to `wall` by `Game.step` (both apply the penalty). On the very first
step of an episode, this can flip `done = True` immediately if the
car's spawn box is on the edge.

## Why it matters

- Subtle reproducibility issue depending on random control-point
  layout.
- Confusing semantics — "out of bounds" is not "out of bounds of the
  track", it's "out of bounds of the **drawing**".

## Fix

Drop the bounding-box early-exit, **or** expand it generously
(e.g. `+ track_width`) so the spawn point is never on the boundary.
The spatial grid already gives O(1) collision checks against actual
segments, so the bbox check provides ~nothing.

```python
# Either remove the block entirely, or:
margin = self.track_width + radius
if (x < self._cached_min_x - margin or ...):
    return True, {'type': 'out_of_bounds'}
```

## Acceptance criteria

- Spawning at any waypoint of any randomly generated track does not
  trigger collision on the first step.
- Driving far off the visible track still triggers collision via the
  wall path (`type: 'wall'`).

## Files touched

- `game/track.py`

## Depends on

— (none)
