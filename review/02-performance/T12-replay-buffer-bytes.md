# T12 — Replay buffer stores states as `bytes` then reparses every sample 🟡

## Problem

`ReplayBuffer.push`:

```python
state_bytes = state.astype(np.float32).tobytes()
next_state_bytes = next_state.astype(np.float32).tobytes()
```

and `sample`:

```python
state = np.frombuffer(transition.state, dtype=np.float32).reshape(1, -1)
```

Reasons this is bad:
- `np.frombuffer` returns a **read-only** view; subsequent operations
  may force a copy anyway.
- Converting to bytes loses dtype/shape metadata — if `STATE_DIM`
  changes you get silent misalignment.
- The supposed benefit (memory savings) is illusory: a 10-float
  float32 vector is 40 bytes either way.

## Fix

Use preallocated numpy arrays:

```python
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dtype=np.int64):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=action_dtype)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        self.states[self.pos]      = s
        self.next_states[self.pos] = ns
        self.actions[self.pos]     = a
        self.rewards[self.pos]     = r
        self.dones[self.pos]       = d
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])
```

This is what every reference DQN implementation does. Bonus: torch
tensors can wrap these arrays with `torch.from_numpy(...).to(device,
non_blocking=True)` in one call.

## Acceptance criteria

- Same agent training results (within RNG noise) before/after the
  swap.
- Microbenchmark: 1000 samples in < 50 ms on CPU.

## Files touched

- `rl/memory.py`
- `rl/agent.py` (constructor — pass `state_dim`)

## Depends on

T08 (do them together — both rewrite memory).
