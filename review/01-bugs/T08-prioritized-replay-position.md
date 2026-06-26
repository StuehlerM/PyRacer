# T08 — `PrioritizedReplayBuffer.pos` is never incremented 🟡

## Problem

`rl/memory.py`:

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        ...
        self.pos = 0  # Current position in buffer
    def push(self, ...):
        self.buffer.append(transition)
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
```

- `self.pos` is set but never used — dead state.
- Indices returned from `sample()` are positions in the current `deque`,
  but after the deque rolls (capacity reached) those indices point to
  *different* transitions next time. Caller has no way to know.
- `replace=False` in `np.random.choice` may fail when probabilities
  collapse near zero.

## Why it matters

- The buffer "works" but is unsafe — if anyone wires it into the agent
  (see T22), priority updates will smear across unrelated transitions
  after capacity is hit.

## Fix

Use a stable index scheme (e.g. a monotonic counter modulo capacity)
or follow the canonical OpenAI Baselines implementation with a
`SumTree`. The simplest correct fix:

1. Use `list` not `deque` so indices stay stable until you overwrite.
2. Track `self.pos` and overwrite explicitly:
   ```python
   if len(self.buffer) < self.capacity:
       self.buffer.append(transition)
       self.priorities.append(max_priority)
   else:
       self.buffer[self.pos] = transition
       self.priorities[self.pos] = max_priority
   self.pos = (self.pos + 1) % self.capacity
   ```
3. Return `indices` exactly as the integers used; callers can pass them
   back to `update_priorities` safely.

## Acceptance criteria

- `pytest -k prioritized` round-trip test: push N>capacity transitions,
  sample, update priorities — assert priorities of *those exact*
  transitions changed and no others.

## Files touched

- `rl/memory.py`

## Depends on

— (lands before T22)
