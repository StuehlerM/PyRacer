# T24 — `car_angle` should be encoded as (sin, cos), not a raw radian 🟡

## Problem

`Game._get_state`:

```python
car_angle = self.car.angle  # Keep as is for now
...
state = np.concatenate([sensor_readings, [car_speed], [car_angle], [progress]])
```

`car.angle` is unbounded (the modulo to `[-π, π]` isn't applied
anywhere), and even if it were, the network sees a discontinuity at
±π that doesn't reflect reality — angles −3.13 rad and +3.13 rad are
~equal in *direction*.

## Why it matters

- The network has to learn to "wrap" a discontinuous feature.
- After enough drifting (T04), values can be 10, 20, 50 rad.

## Fix

1. Wrap the angle in `Car.update`:
   ```python
   self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
   ```
2. Encode as two features:
   ```python
   state = np.concatenate([
       sensor_readings,
       [car_speed, np.sin(car_angle), np.cos(car_angle), progress]
   ])
   ```
3. Update `config.STATE_DIM = NUM_SENSORS + 4`.
4. Saved models become incompatible — bump model version. Add a
   `state_version` field in the saved checkpoint and refuse to load
   mismatched versions.

Bonus: consider whether the absolute angle is even useful. In a turn,
what the agent cares about is *relative to track tangent*. Switching
to "heading error from track tangent" (a single signed angle) is a
common and strong improvement — but bigger change.

## Acceptance criteria

- Symbol `STATE_DIM` updated and every consumer reflects it.
- Old checkpoints fail with a clear error message ("trained with
  state_version=1, but code expects state_version=2").
- New checkpoints train successfully.

## Files touched

- `game/game.py`, `game/car.py`
- `utils/config.py`
- `rl/agent.py` (checkpoint metadata)

## Depends on

T04, T20.
