# T22 — `PrioritizedReplayBuffer` exists but is unreachable 🟡

## Problem

`rl/memory.py` defines `PrioritizedReplayBuffer`, but
`rl/__init__.py` doesn't export it, and `DQNAgent` always instantiates
`ReplayBuffer`.

## Why it matters

- Dead code — extra surface area, no benefit.
- Removes a useful experiment (PER often helps DQN convergence).

## Fix

1. Land T08 so the implementation is sound.
2. Add `use_prioritized: bool = False` to `DQNAgent.__init__`.
3. Switch buffer class accordingly:
   ```python
   if use_prioritized:
       self.memory = PrioritizedReplayBuffer(memory_size, ...)
       self._uses_per = True
   else:
       self.memory = ReplayBuffer(memory_size, state_dim=state_dim)
       self._uses_per = False
   ```
4. In `update()`, if PER is in use, unpack `(states, actions, ...,
   indices, weights)` and multiply the per-element loss by `weights`:
   ```python
   td_errors = current_q - target_q
   loss = (weights * td_errors.pow(2)).mean()
   self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy() + 1e-6)
   ```
5. Add `--prioritized` flag to `train.py`.

## Acceptance criteria

- `python train.py --prioritized --episodes 10` runs without errors
  and produces non-zero loss.
- `python -m pytest` includes a smoke test of the PER path.

## Files touched

- `rl/agent.py`, `rl/memory.py`, `rl/__init__.py`
- `train.py`

## Depends on

T08 (fixes the buffer), T12 (storage strategy).
