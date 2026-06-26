# T03 — Steering-return logic double-applies sign and ignores `dt` correctly 🟠

## Problem

`Car.update` "return to centre" branch reads:

```python
return_amount = -np.sign(self.steering_angle) * self.steering_return_speed * dt
max_return    = abs(self.steering_angle)
return_amount = np.clip(abs(return_amount), 0, max_return) * -np.sign(self.steering_angle)
self.steering_angle = clamp(self.steering_angle + return_amount, ...)
```

The first computation already has the correct sign. Taking `abs(...)`
then re-applying `-np.sign(...)` is harmless mathematically but
indicates the code drifted — and the `np.clip(...).item()` returns a
numpy scalar that is implicitly broadcast back to float, which is fine
but slow.

More importantly: at high frame rates the deadzone snap-to-zero
(`abs(...) < deg2rad(0.5)`) can mask the case where the return-speed is
larger than the current angle, which causes a *jitter* near zero
because the clamp overshoots and then snaps.

## Why it matters

- Steering feel changes with frame rate / `dt`.
- Logic is hard to read and easy to break.
- Minor: small per-step performance penalty from redundant `abs`+`sign`.

## Fix

Rewrite the branch cleanly:

```python
if abs(steering) < 1e-3:
    # Return to centre at most `steering_return_speed * dt` per tick.
    max_step  = self.steering_return_speed * dt
    step_size = min(max_step, abs(self.steering_angle))
    self.steering_angle -= np.sign(self.steering_angle) * step_size
    if abs(self.steering_angle) < 1e-4:
        self.steering_angle = 0.0
```

This is frame-rate-independent, has no double sign, and the deadzone
is only used for the *final* snap.

## Acceptance criteria

- Manually driving and releasing the steering keys returns wheel to
  centre smoothly with no visible jitter at any FPS.
- Holding steering for N ticks and releasing for the same N ticks
  returns the angle to zero within one tick.

## Files touched

- `game/car.py`

## Depends on

— (none)
