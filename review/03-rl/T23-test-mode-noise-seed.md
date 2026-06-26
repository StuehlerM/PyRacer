# T23 — Runs are not reproducible (no seeding controls) 🟡

## Problem

Random sources used in the project:

- `np.random` for control points (`Track._generate_track`)
- `random` and `np.random` for action selection (epsilon-greedy)
- `torch` for network init and dropout (no dropout, but init counts)
- `random.sample` in `ReplayBuffer.sample`

None of these are seeded anywhere. The same command line produces a
different track and a different policy every time.

## Why it matters

- Impossible to A/B test changes.
- Bug repros depend on luck.
- AGENTS.md *suggests* a pattern (`random.seed(42)`) but no code uses
  it.

## Fix

Add `--seed N` to both `train.py` and `test.py`. In `main()`:

```python
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if args.seed is not None:
    set_seed(args.seed)
```

For full reproducibility on CUDA also set:
```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

…but document that determinism trades for speed.

Also: change `Track.__init__` to accept a `seed` argument so a single
track shape can be reproduced regardless of order of operations.

## Acceptance criteria

- `python train.py --seed 42 --episodes 5` produces an identical CSV
  on two consecutive runs (modulo CUDA nondeterminism, document the
  caveat).

## Files touched

- `train.py`, `test.py`
- `game/track.py`
- `utils/config.py` (optional default seed)

## Depends on

— (none)
