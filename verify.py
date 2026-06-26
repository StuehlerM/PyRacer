#!/usr/bin/env python3
"""
Verification script to check if the project is set up correctly.
"""

import sys


def verify_python_version():
    """Verify Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [ERROR] Python 3.8 or higher is required")
        return False
    print("  [OK]")
    return True


def verify_imports():
    """Verify required packages can be imported."""
    print("\nChecking required packages...")
    
    required = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'pygame': 'Pygame'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  {name}: [OK]")
        except ImportError:
            print(f"  {name}: [NOT INSTALLED]")
            all_ok = False
    
    return all_ok


def verify_project_files():
    """Verify project files exist."""
    print("\nChecking project files...")
    
    import os
    
    required_files = [
        'game/car.py',
        'game/track.py',
        'game/game.py',
        'game/physics.py',
        'game/__init__.py',
        'rl/agent.py',
        'rl/model.py',
        'rl/memory.py',
        'rl/environment.py',
        'rl/__init__.py',
        'utils/config.py',
        'utils/__init__.py',
        'main.py',
        'train.py',
        'test.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  {file}: [OK]")
        else:
            print(f"  {file}: [MISSING]")
            all_ok = False
    
    return all_ok


def verify_game_components():
    """Verify game components can be imported (without pygame rendering)."""
    print("\nChecking game components...")
    
    try:
        # Test physics (no pygame dependency)
        from game.physics import line_segment_intersection, point_to_line_distance
        print("  Physics module: [OK]")
        
        # Test config
        from utils.config import config
        print("  Config module: [OK]")
        print(f"    - State dimension: {config.STATE_DIM}")
        print(f"    - Action dimension: {config.ACTION_DIM}")
        print(f"    - Screen size: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
        
        # Test models (no pygame dependency)
        from rl.model import DQN
        print("  DQN model: [OK]")
        
        # Test memory
        from rl.memory import ReplayBuffer
        print("  Replay buffer: [OK]")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PyRacer - Project Verification")
    print("=" * 60)
    
    results = []
    
    # Run checks
    results.append(("Python Version", verify_python_version()))
    results.append(("Required Packages", verify_imports()))
    results.append(("Project Files", verify_project_files()))
    results.append(("Game Components", verify_game_components()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All checks passed! Project is ready to use.")
        print("\nTo get started:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Play the game: python main.py")
        print("  3. Train the agent: python train.py --episodes 100 --render")
        return 0
    else:
        print("\n[FAILED] Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
