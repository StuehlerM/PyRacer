import pytest
import numpy as np

from game.game import Game
from game.track import Track
from utils.config import config


def _make_headless_game(seed=0):
    game = Game(track=Track(seed=seed), headless=True)
    game.reset()
    return game


def test_forward_spline_progress_to_next_checkpoint_is_rewarded(monkeypatch):
    game = _make_headless_game()
    game.prev_progress = 0.10

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (0.11, 0.0) if return_distance else 0.11
        ),
    )

    _, reward, done, info = game.step(0)
    progress_delta = 0.01
    expected = progress_delta * config.REWARD_CHECKPOINT_APPROACH
    assert reward == pytest.approx(expected)
    assert not done
    assert info["off_track"] is False
    assert info["collision"] is False


def test_track_aligned_forward_velocity_is_rewarded(monkeypatch):
    game = _make_headless_game()
    game.car.velocity = np.array([game.car.max_speed * 0.5, 0.0])

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(game.track, "get_direction_at_progress", lambda progress: np.array([1.0, 0.0]))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (game.prev_progress, 0.0) if return_distance else game.prev_progress
        ),
    )

    _, reward, done, info = game.step(0)

    forward_reward = 0.5 * config.REWARD_FORWARD_SPEED
    expected = config.REWARD_INITIAL + forward_reward
    assert reward == pytest.approx(expected)
    assert not done
    assert info["forward_drive_reward"] == pytest.approx(forward_reward)


def test_off_track_applies_grace_penalty_without_terminating(monkeypatch):
    game = _make_headless_game()

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (game.prev_progress, game.track.track_width) if return_distance else game.prev_progress
        ),
    )

    # A single step off the road is survivable: the car pays the per-step off-track
    # penalty (plus the idle time penalty) but the run keeps going so it can recover.
    _, reward, done, info = game.step(0)
    expected = config.REWARD_OFF_TRACK + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert not done
    assert info["off_track"] is True
    assert info["collision"] is False
    assert info["steps_off_track"] == 1


def test_off_track_too_long_terminates_as_crash(monkeypatch):
    game = _make_headless_game()

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (game.prev_progress, game.track.track_width) if return_distance else game.prev_progress
        ),
    )

    # Staying off the road past the grace window ends the run as a crash. The
    # off-track patience (90 steps) is shorter than the stall patience (240), so
    # the off-track timeout is what terminates here.
    done = False
    reward = 0.0
    info = {}
    for _ in range(config.OFF_TRACK_PATIENCE_STEPS):
        _, reward, done, info = game.step(0)

    expected = (
        config.REWARD_OFF_TRACK + config.REWARD_COLLISION + config.REWARD_TIME_PENALTY
    )
    assert reward == pytest.approx(expected)
    assert done is True
    assert info["off_track"] is True
    assert info["collision"] is True
    assert info["stalled"] is False
    assert info["steps_off_track"] == config.OFF_TRACK_PATIENCE_STEPS


def test_no_progress_adds_penalty_and_terminates(monkeypatch):
    game = _make_headless_game()

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (game.prev_progress, 0.0) if return_distance else game.prev_progress
        ),
    )

    done = False
    reward = 0.0
    info = {}
    for _ in range(config.NO_PROGRESS_PATIENCE_STEPS):
        _, reward, done, info = game.step(0)

    expected = config.REWARD_NO_PROGRESS + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert done
    assert info["stalled"] is True


def test_moving_away_from_next_checkpoint_is_penalized(monkeypatch):
    game = _make_headless_game()
    game.prev_progress = 0.20

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (0.19, 0.0) if return_distance else 0.19
        ),
    )

    _, reward, done, info = game.step(0)

    progress_delta = -0.01
    expected = (
        progress_delta * config.REWARD_CHECKPOINT_APPROACH * config.REWARD_WRONG_WAY_MULTIPLIER
        + config.REWARD_TIME_PENALTY
    )
    assert reward == pytest.approx(expected)
    assert not done
    assert info["forward_progress"] == pytest.approx(0.0)
