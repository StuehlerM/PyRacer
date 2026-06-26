# T06 — Checkpoint `reached` flags survive `reset()` 🔴

## Problem

`Track._create_checkpoints()` builds:

```python
self.checkpoints.append({
    'position': checkpoint_pos,
    'index': i,
    'reached': False,    # never read; instead the game uses last_checkpoint_idx
})
```

The game then uses `self.last_checkpoint_idx` (an int) and `Game.reset()`
correctly resets it to `-1`. Good.

**But** in `MultiTrackEnv`, when the env swaps tracks (`set_track`), the
new track's checkpoint dicts may already have stale data (currently
they're freshly generated, so this is okay), **and** when the same
`Track` instance is reused across episodes (the default), `start_idx`
in `_generate_track` is computed once and never recomputed even though
control points are re-randomised on `Track.__init__`.

The real bug: the `'reached'` field on checkpoint dicts is never used,
so it's just dead code. The actual checkpoint-completion logic is
correct **but** there's a second problem hiding here:

`Track.check_checkpoint()` only checks checkpoints whose `index >
last_checkpoint_idx`. After completing one full lap, `last_checkpoint_idx`
becomes the finish-line index (e.g. `4`). On the next lap, the agent
needs to pass checkpoint 0 again — but its index is `0 < 4`, so it is
never reported. **Only the first lap of any episode produces lap
rewards.**

## Why it matters

- Lap-2 and later never reward the agent.
- Multi-lap training is silently broken (which is mostly hidden because
  episodes usually end on collision first).

## Fix

Reset `last_checkpoint_idx` to `-1` whenever a finish-line checkpoint is
crossed:

```python
# In Game.step at lap completion:
self.last_checkpoint_idx = -1   # restart checkpoint sequence for next lap
```

Optionally also remove the unused `'reached'` field from
`Track._create_checkpoints` to avoid confusion.

## Acceptance criteria

- Drive the car through two consecutive laps; the second lap also
  fires checkpoint rewards and increments `lap_count` to 2.
- Unit test: simulate sequential checkpoints, assert `lap_count == n`
  after `n` simulated laps.

## Files touched

- `game/game.py`
- `game/track.py` (optional cleanup of `'reached'`)

## Depends on

T01, T02 (same code region).
