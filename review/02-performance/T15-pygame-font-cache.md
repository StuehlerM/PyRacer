# T15 — `pygame.font.SysFont` is recreated every frame 🟢

## Problem

`Game._draw_hud` and the pause render path do:

```python
font = pygame.font.SysFont(None, 24)
```

every single frame. `SysFont` discovery is slow on Windows (tens of ms
on cold cache).

## Why it matters

- Frame time jitter, especially first-paint after pause.
- Trivial fix.

## Fix

Create fonts once in `Game.__init__`:

```python
if not headless:
    pygame.font.init()
    self._font_hud   = pygame.font.SysFont(None, 24)
    self._font_pause = pygame.font.SysFont(None, 72)
```

Use them in `_draw_hud` and `run` instead of constructing locally.

## Acceptance criteria

- HUD still draws.
- No `SysFont` call inside render loop (`grep_search` confirms).

## Files touched

- `game/game.py`

## Depends on

— (none)
