#!/usr/bin/env python3
"""
Main entry point for human play mode.
Run this to play the game manually with keyboard controls.

Controls:
- UP/W: Accelerate
- DOWN/S: Brake/Reverse
- LEFT/A: Turn left
- RIGHT/D: Turn right
- R: Reset car position
- S: Toggle sensor visualization
- SPACE: Pause
- ESC: Quit
"""

from game.game import Game
from game.track import Track
from utils.config import config


def main():
    # Human-play entry point: shares same game core as training/testing, but skips RL loop.
    print("PyRacer - 2D Racing Game")
    print("========================")
    print("\nControls:")
    print("  UP: Accelerate")
    print("  DOWN: Brake/Reverse")
    print("  LEFT: Turn left")
    print("  RIGHT: Turn right")
    print("  R: Reset car")
    print("  L: Toggle Learning/Human mode")
    print("  S (hold): Show sensor rays")
    print("  SPACE: Pause")
    print("  ESC: Quit")
    print("\nNote: Use ONLY arrow keys for driving.")
    print("Letter keys (WASD) are free for future features.")
    print("\nPress ESC to exit...")
    
    # Main loop lives inside Game so keyboard play and RL modes use one simulation code path.
    game = Game(headless=False)
    
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    finally:
        game.close()


if __name__ == "__main__":
    main()
