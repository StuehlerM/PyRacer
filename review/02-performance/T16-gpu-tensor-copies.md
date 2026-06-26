# T16 — Per-step tensor allocations cause excess host→device copies 🟢

## Problem

`DQNAgent._greedy_action`:

```python
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
q_values = self.policy_net(state_tensor)
```

Every env step on GPU does:
1. Allocate a CPU tensor from numpy.
2. Copy 10 floats to GPU.
3. Allocate output, copy back.

That's ~10–20 µs of overhead per call. At 60 Hz × 2000 steps × 10 000
episodes that's ~3.3 hours of pure overhead at typical rates.

Also: `torch.FloatTensor(state)` is deprecated; prefer
`torch.from_numpy(state).float()`.

## Why it matters

- Mostly matters when actually using CUDA. CPU training feels it less.
- Easy to fix.

## Fix

1. Reuse a preallocated tensor (`pinned=True` if GPU):
   ```python
   self._inference_buf = torch.zeros((1, state_dim), device=self.device, dtype=torch.float32)
   ```
   Then:
   ```python
   self._inference_buf[0].copy_(torch.from_numpy(state), non_blocking=True)
   q = self.policy_net(self._inference_buf)
   ```
2. Wrap inference in `with torch.inference_mode():` instead of
   `no_grad()` (marginally faster).
3. In `update()`, build batch tensors with
   `torch.from_numpy(arr).to(self.device, non_blocking=True)` rather
   than `FloatTensor(arr).to(...)`.

## Acceptance criteria

- No deprecation warnings about `FloatTensor` during training.
- On CUDA: env step throughput visibly increases (measure with
  `--episodes 50` timing).

## Files touched

- `rl/agent.py`

## Depends on

T12 (memory rewrite makes tensor conversion cleaner).
