# T04 — Car drift logic can override `start_angle` immediately on reset 🟠

## Problem

`Car.update` continuously aligns `self.angle` to the velocity vector:

```python
if self.speed > 1.0:
    target_angle = np.arctan2(self.velocity[1], self.velocity[0])
    angle_diff = (target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
    self.angle += angle_diff * 0.1
```

On the *first* step after `reset`, `self.velocity` is `[0, 0]` because
`reset()` zeroes it. That branch is gated by `self.speed > 1.0`, so it's
fine for one step. But once the agent applies any throttle, the
velocity vector is computed using `angle + steering_angle`, *not* the
plain `angle`, so the auto-align snaps `self.angle` toward
`angle + steering_angle` — meaning even a momentary turning input
"bakes in" a rotation on the car body itself.

Net effect: after a few frames of turning, the car body is rotated such
that the *sensors* point in a different direction than the actual
direction of travel — they look at the boundary they're driving
*through*, not the one ahead.

## Why it matters

- Sensor readings become misleading during turns.
- Reinforcement learning gets nonstationary state observations for the
  "same" turning manoeuvre.
- Hard to debug because the visual indicator (red dot on car front)
  *also* follows `self.angle`, so it looks correct on screen.

## Fix

Track *heading* separately from *body orientation*, or — simpler — stop
auto-rotating the body. The cleanest one-liner:

```python
# Remove the drift block entirely; use angle as the car's heading.
# The wheel steering effect is already applied via effective_angle
# when computing velocity.
```

If you want a drifting visual effect, render the body at
`heading + small_drift_offset` where `small_drift_offset` is derived
from lateral velocity, but **never** write back to `self.angle`.

## Acceptance criteria

- Driving a full circle returns `self.angle` to its starting value
  ± numerical noise (no integrated drift).
- Sensor rays remain symmetric around the direction of motion when
  driving straight at any speed.

## Files touched

- `game/car.py`

## Depends on

— (none)
