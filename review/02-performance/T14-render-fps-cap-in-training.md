# T14 — `clock.tick(FPS)` is called inside the render path during training 🟡

## Problem

`Game.render` ends with `self.clock.tick(config.FPS)` — capping the
loop to 60 Hz. When `train.py --render` is used, the trainer also
manually sleeps `time.sleep(0.01)` after each step. Both caps stack and
limit training throughput unnecessarily.

When `headless=True` (the default for training), `render()` returns
early, so this isn't an issue. **But** `Game.step` *unconditionally*
calls `self.render(...)` at the end if `not self.headless` — meaning
*any* graphical training run is throttled to 60 Hz of *physics*, not
60 Hz of *display*.

## Why it matters

- `train.py --render` is ~60× slower than necessary on modern
  hardware.
- The `time.sleep(0.01)` in `train.py` and `test.py` is in addition to
  the FPS cap, so users get an even worse experience.

## Fix

- Remove `time.sleep(0.01)` from `train.py` and `test.py`.
- Decouple physics from rendering: in `Game.step`, only call `render()`
  when rendering is requested AND throttling is desired:

  ```python
  if not self.headless and self.render_every_step:
      self.render(...)
  ```

- Add `render_every_n` to skip frames: `render_every_n=4` shows the
  agent at 15 fps but trains 4× as many env steps.

- The FPS cap belongs in the *human* loop (`run()`), not in `render()`.
  Move `self.clock.tick(config.FPS)` from `render` into `run` only.

## Acceptance criteria

- `python train.py --render --episodes 10` finishes noticeably faster
  than before.
- `python main.py` (human play) still runs at exactly 60 fps.

## Files touched

- `game/game.py`
- `train.py`, `test.py`

## Depends on

T28 (renders coupling is fixed there too).
