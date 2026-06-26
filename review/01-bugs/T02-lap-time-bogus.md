# T02 — `lap_time` is logged as 0.0167 s on lap completion 🔴

## Problem

In `Game.step`, when a lap is completed:

```python
current_lap_time = self.lap_time
...
if current_lap_time < self.best_lap_time:
    self.best_lap_time = current_lap_time
self.lap_time = 0.0      # reset BEFORE we increment below
...
self.lap_time += config.DT   # adds 1/60 = 0.0167 s
```

Because `self.lap_time` is reset to 0 inside the checkpoint branch and
then `+= DT` runs unconditionally at the bottom of `step`, the
`info['lap_time']` returned to the trainer on the lap-completion step is
`0` and **the next** observed lap time floor is `0.0167 s`.

The training CSV confirms this — `lap_time` and `best_lap_time` are
literally `0.01666...` for every episode that ever crosses the finish.

## Why it matters

- `best_lap_time` is meaningless — every run looks like a world record.
- The "save best model" logic (T19) fires constantly because the
  threshold is permanently at `0.0167`.
- Plots / comparisons of lap times are noise.

## Fix

Capture the lap time **before** resetting and return that value via
`info`. Move the `self.lap_time = 0.0` reset to happen *after* the
returned value is computed:

```python
# Inside step(), at lap completion:
current_lap_time = self.lap_time
if current_lap_time < self.best_lap_time:
    self.best_lap_time = current_lap_time
    new_best = True
else:
    new_best = False
self.lap_count += 1
self._completed_lap_time = current_lap_time   # remember for info dict
self.lap_time = 0.0                            # reset *after* recording

...
info = {
    ...
    'lap_time': self._completed_lap_time if is_finish else self.lap_time,
    'lap_completed': is_finish,
    'new_best_lap': is_finish and new_best,
}
```

Then at the *top* of `step`, increment `self.lap_time` instead of at
the bottom — that makes the relationship to physics ticks easier to
reason about and removes the off-by-one.

## Acceptance criteria

- After a few hundred random episodes, `best_lap_time` is a realistic
  number (multiple seconds), not `0.0167`.
- `info['lap_time']` on the finish-line step is the elapsed time for
  that lap, not zero or `DT`.

## Files touched

- `game/game.py`

## Depends on

T01 (touching the same code region — rebase on top of T01).
