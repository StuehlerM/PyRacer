# PyRacer Code Review — Improvement Plan

> Generated: 2026-06-20
> Reviewer: GitHub Copilot (Anthropic: Claude Opus 4.7)
> Scope: Full project review of `game/`, `rl/`, `utils/`, entry-point scripts, and supporting files.

This folder contains a code review and prioritized improvement plan for PyRacer.
Every issue has been broken into a **small, independently-implementable task**
so they can be picked up one at a time, reviewed in isolation, and shipped
without blocking other work.

---

## How to use this plan

1. Pick a task file (e.g. `01-bugs/T01-reward-double-counting.md`).
2. Read the **Problem**, **Why it matters**, and **Fix** sections.
3. Implement and verify against the **Acceptance criteria**.
4. Tick it off in the table below.

Each task is self-contained — you do **not** need to do them in order
unless a `Depends on:` line says otherwise.

---

## Severity legend

| Symbol | Meaning |
|---|---|
| 🔴 | **Critical** — Bug that produces incorrect behaviour or crashes |
| 🟠 | **High** — Wrong/misleading metric, broken feature, or significant footgun |
| 🟡 | **Medium** — Quality / correctness issue worth fixing soon |
| 🟢 | **Low** — Polish, style, or nice-to-have |

---

## Task index

### 01 — Bugs (correctness)
| ID | Severity | Title |
|---|---|---|
| [T01](01-bugs/T01-reward-double-counting.md) | 🔴 | Episode reward is double-counted in `Game.step` |
| [T02](01-bugs/T02-lap-time-bogus.md) | 🔴 | `lap_time` is logged as 0.0167 s because reset order is wrong |
| [T03](01-bugs/T03-steering-return-frame-rate-dependent.md) | 🟠 | Steering-return logic ignores `dt` and double-applies sign |
| [T04](01-bugs/T04-car-angle-not-reset.md) | 🟠 | `Car.reset` ignores the `Track.start_angle` because of missing assignment edge cases |
| [T05](01-bugs/T05-progress-wraparound.md) | 🟠 | Lap completion broken when progress wraps from ~1.0 back to 0 |
| [T06](01-bugs/T06-checkpoint-reset-missing.md) | 🔴 | Checkpoint `reached` flag never reset between episodes |
| [T07](01-bugs/T07-out-of-bounds-collision.md) | 🟠 | `check_collision` returns `out_of_bounds` even when car is inside the loop |
| [T08](01-bugs/T08-prioritized-replay-position.md) | 🟡 | `PrioritizedReplayBuffer.pos` is never incremented and indices drift |
| [T09](01-bugs/T09-test-results-progress-bug.md) | 🟡 | `_save_results` in `test.py` writes nonsense for progress & collisions |
| [T10](01-bugs/T10-multi-track-test-confusion.md) | 🟡 | `MultiTrackEnv` only rotates tracks every 100 episodes (silent on test) |

### 02 — Performance
| ID | Severity | Title |
|---|---|---|
| [T11](02-performance/T11-ray-cast-vectorize.md) | 🟠 | Vectorise `ray_cast` and `line_segment_intersection` with NumPy |
| [T12](02-performance/T12-replay-buffer-bytes.md) | 🟡 | Replay buffer stores states as `bytes` then re-decodes every sample |
| [T13](02-performance/T13-progress-search-localize.md) | 🟡 | `Track.get_progress` does a full O(N) waypoint scan every step |
| [T14](02-performance/T14-render-fps-cap-in-training.md) | 🟡 | `clock.tick(FPS)` runs in render path even during fast training |
| [T15](02-performance/T15-pygame-font-cache.md) | 🟢 | `pygame.font.SysFont` is recreated every frame in `_draw_hud` |
| [T16](02-performance/T16-gpu-tensor-copies.md) | 🟢 | Per-step tensor creation in `select_action` causes excess host→device copies |

### 03 — RL correctness & training quality
| ID | Severity | Title |
|---|---|---|
| [T17](03-rl/T17-epsilon-decay-per-update.md) | 🟠 | Epsilon decays only on training updates → resumed runs decay too slowly |
| [T18](03-rl/T18-train-start-episode-unused.md) | 🟠 | `TRAIN_START_EPISODE` is configured but never applied |
| [T19](03-rl/T19-best-model-trigger.md) | 🟠 | "Best model" is saved on first lap completion only, regardless of quality |
| [T20](03-rl/T20-state-normalization.md) | 🟡 | `RacingEnv._normalize_state` is a no-op despite the docstring |
| [T21](03-rl/T21-reward-shaping-progress.md) | 🟡 | Progress reward swings to ~+100 on lap wrap and dominates the signal |
| [T22](03-rl/T22-prioritized-replay-wired-up.md) | 🟡 | `PrioritizedReplayBuffer` exists but is never used by `DQNAgent` |
| [T23](03-rl/T23-test-mode-noise-seed.md) | 🟡 | No seeding controls — runs are not reproducible |
| [T24](03-rl/T24-state-action-encoding.md) | 🟡 | Car angle stored unbounded; should be encoded as (sin, cos) |
| [T25](03-rl/T25-target-update-double-step.md) | 🟡 | Target net updates every 10 *training* steps, not env steps as configured |

### 04 — Architecture & code quality
| ID | Severity | Title |
|---|---|---|
| [T26](04-architecture/T26-config-mutable-singleton.md) | 🟡 | `config` is a mutable class-level singleton — CLI args don't propagate |
| [T27](04-architecture/T27-game-step-action-types.md) | 🟡 | `Game.step` accepts both ints and tuples — split into clear paths |
| [T28](04-architecture/T28-render-coupled-with-step.md) | 🟡 | `render()` is invoked inside `step()` — breaks headless training cleanliness |
| [T29](04-architecture/T29-circular-package-imports.md) | 🟡 | `game/__init__.py` imports everything, slowing imports & risking cycles |
| [T30](04-architecture/T30-magic-numbers.md) | 🟢 | Magic numbers (50, 0.8, 0.1, `0.005`) scattered — move to `Config` |
| [T31](04-architecture/T31-logging-vs-print.md) | 🟢 | Mix of `print()` and `logging` — pick one |
| [T32](04-architecture/T32-empty-verify-new.md) | 🟢 | `verify_new.py` is empty — remove or implement |
| [T33](04-architecture/T33-game-init-pygame-display.md) | 🟡 | `pygame.init()` + display always created — blocks pure-RL CI |

### 05 — Testing & verification
| ID | Severity | Title |
|---|---|---|
| [T34](05-testing/T34-pytest-suite.md) | 🟠 | No real test suite — convert `test_model()` / `test_agent()` into pytest |
| [T35](05-testing/T35-physics-property-tests.md) | 🟡 | Add property-based tests for `line_segment_intersection`, `ray_cast` |
| [T36](05-testing/T36-smoke-test-training.md) | 🟡 | Add a 10-episode smoke training test with deterministic seed |
| [T37](05-testing/T37-ci-pipeline.md) | 🟢 | No CI configuration — add GitHub Actions workflow |

### 06 — Documentation
| ID | Severity | Title |
|---|---|---|
| [T38](06-docs/T38-readme-vs-truth.md) | 🟡 | README claims features (`set_track`, `--log-freq`) that aren't quite there |
| [T39](06-docs/T39-agents-md-outdated.md) | 🟢 | `AGENTS.md` reward-function snippet doesn't match current code |
| [T40](06-docs/T40-state-vector-doc.md) | 🟢 | Document state vector layout in one canonical place |

### 07 — DX / tooling
| ID | Severity | Title |
|---|---|---|
| [T41](07-dx/T41-cli-args-respected.md) | 🟡 | Several CLI args (`--lr`, `--gamma`) ignored after first run |
| [T42](07-dx/T42-tensorboard-not-wired.md) | 🟢 | `tensorboard` in requirements but no `SummaryWriter` anywhere |
| [T43](07-dx/T43-saved-models-glob.md) | 🟢 | No "load latest model" helper — users must paste timestamp manually |
| [T44](07-dx/T44-python-version-pin.md) | 🟢 | `pygame-ce` / `torch` on Python 3.14 may have no wheels — pin `<3.13` |

---

## Suggested order of attack

If you want a short critical path:

1. **T01, T02, T06** — fix reward / lap-time / checkpoint resetting (rewards are currently being computed against broken data).
2. **T17, T18, T19** — three small RL training fixes that change learning curves dramatically.
3. **T26, T41** — make CLI args actually take effect.
4. **T34** — add a real pytest skeleton so subsequent fixes can be verified mechanically.
5. Everything else — pick by severity or by what you happen to be touching.

---

## Notes & non-issues

A few things looked suspicious at first but are actually fine:

- `F` is imported in `rl/agent.py` (line 10). The traceback in `training_log.txt` is from a prior commit before that import was added — current code runs.
- `pygame-ce` on Python 3.14 worked locally per the same log, but be aware that 3.14 wheels for `torch` and `pygame-ce` lag releases (see T44).
- Spatial grid (`Track._build_spatial_grid`) is well-designed and a real win — keep it.
- Double DQN + Polyak averaging in `DQNAgent` is correctly implemented.
