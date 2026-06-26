# T20 — `RacingEnv._normalize_state` is a no-op despite the docstring 🟡

## Problem

```python
def _normalize_state(self, state):
    """Normalize the state vector..."""
    # For now, we don't do online normalization
    return state
```

`_update_stats` partially maintains `state_means` / `state_counts` (no
variance), and neither is read anywhere. So:

- The agent receives `car_angle` unbounded (`[-π, π]` but accumulates
  numerical drift — see T04 and T24).
- `progress` is `[0, 1]` — fine.
- `car_speed` is normalized to `[-1, 1]` — fine.
- `sensor_readings` are `[0, 1]` — fine.

So the only un-normalised component is **angle**. That alone needs
attention (T24), but the misleading scaffolding here invites a future
bug.

## Why it matters

- Code that pretends to do something is a footgun.
- Online normalisation is mentioned in comments and AGENTS.md, but
  nobody benefits from it.

## Fix

Either:

1. **Delete it.** Remove `_normalize_state`, `_update_stats`,
   `state_means`, `state_stds`, `state_counts`. The state from `Game`
   is already mostly normalized; the angle problem belongs in T24.

2. **Implement it.** Use Welford's algorithm for running mean/std and
   actually return `(state - mean) / (std + 1e-5)`. Persist
   `mean`/`std` in checkpoints so eval matches training.

Recommended: **option 1**, since per-feature normalisation in the env
is fragile and torch can include a `LayerNorm` first if desired.

## Acceptance criteria

- No dead code referencing `state_means` / `state_stds`.
- README / AGENTS.md updated accordingly.

## Files touched

- `rl/environment.py`
- `README.md`, `AGENTS.md`

## Depends on

T24 (angle encoding).
