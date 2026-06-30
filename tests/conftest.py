"""Shared pytest configuration.

Runs the whole suite headless so it works on CI / machines with no display:
the SDL video and audio drivers are forced to the "dummy" backend *before*
pygame is ever imported. Also guarantees the project root is importable
regardless of how pytest is launched.
"""
import os
import sys

# Headless rendering — must be set before pygame is imported anywhere.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Make `import utils`, `import rl`, `import game` work from any invocation.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every RNG before each test so failures are reproducible."""
    import random
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
