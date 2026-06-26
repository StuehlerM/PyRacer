# T11 — Vectorise `ray_cast` / `line_segment_intersection` 🟠

## Problem

`game/physics.py` currently does:

```python
for segment_start, segment_end in segments_to_check:
    intersection, intersects = line_segment_intersection(
        start, ray_end, segment_start, segment_end
    )
    ...
```

For each of the 7 sensors per step, and each candidate segment from the
spatial grid (often 20–60), we call `line_segment_intersection` which
internally converts the four points to numpy arrays. That's
`7 × 60 × 4 = ~1,700` tiny numpy allocations per simulation step at
60 Hz → millions per training run.

Profiling will confirm raycasting + collision are the dominant CPU
cost.

## Why it matters

- Training speed is bottlenecked here, not on the neural net.
- The spatial grid (already in place) makes the *count* small, but the
  per-call overhead is what dominates.

## Fix

Rewrite `ray_cast` to take all candidate segments as a single
`(N, 2, 2)` numpy array and run line/line intersection vectorised:

```python
def ray_cast_batch(start, direction, max_distance, seg_starts, seg_ends):
    """
    start: shape (2,)
    direction: shape (2,)
    seg_starts: shape (N, 2)
    seg_ends:   shape (N, 2)
    """
    ray_end = start + direction * max_distance
    p1 = start[None, :]                # (1, 2)
    p2 = ray_end[None, :]
    p3 = seg_starts
    p4 = seg_ends

    d1 = p2 - p1                        # (1, 2)
    d2 = p4 - p3                        # (N, 2)
    denom = d2[:, 1] * d1[:, 0] - d2[:, 0] * d1[:, 1]
    ...
    # returns the minimum t along the ray, or max_distance
```

Cache `seg_starts` / `seg_ends` once in `Track._cache_track_data()`.
Pre-index by spatial grid as today, then use boolean masks.

Expect a 5–10× speedup of `get_sensor_readings` and `check_collision`.

## Acceptance criteria

- Microbenchmark: 10 000 ray-casts on a typical track in < 200 ms
  on CPU (current: > 1 s).
- Unit test: vectorised result matches scalar `ray_cast` to 1e-6 for
  random rays.

## Files touched

- `game/physics.py`
- `game/track.py` (cache `seg_starts`, `seg_ends`)
- `game/car.py` (use new API in `get_sensor_readings`)

## Depends on

— (none, but easier after T34 testing exists)
