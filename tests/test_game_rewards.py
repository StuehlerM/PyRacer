import pytest
import numpy as np

from game.game import Game
from game.track import Track
from utils.config import config


def _make_headless_game(seed=0):
    game = Game(track=Track(seed=seed), headless=True)
    game.reset()
    return game


def test_closing_distance_to_next_checkpoint_is_rewarded(monkeypatch):
    game = _make_headless_game()
    game.car.position = np.array([90.0, 0.0])
    game.track.checkpoints = [{'index': 0, 'position': (100.0, 0.0)}]
    game.prev_checkpoint_distance = 20.0 / np.hypot(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (0.1, 0.0) if return_distance else 0.1
        ),
    )

    _, reward, done, info = game.step(0)
    distance_delta = 10.0 / np.hypot(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    expected = distance_delta * config.REWARD_CHECKPOINT_APPROACH + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert not done
    assert info["off_track"] is False


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
    expected = config.REWARD_INITIAL + forward_reward + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert not done
    assert info["forward_drive_reward"] == pytest.approx(forward_reward)


def test_off_track_adds_penalty_and_terminates(monkeypatch):
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

    _, reward, done, info = game.step(0)
    expected = config.REWARD_OFF_TRACK + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert done is config.OFF_TRACK_TERMINATES
    assert info["off_track"] is True


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
    game.car.position = np.array([80.0, 0.0])
    game.track.checkpoints = [{'index': 0, 'position': (100.0, 0.0)}]
    game.prev_checkpoint_distance = 10.0 / np.hypot(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

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

    _, reward, done, info = game.step(0)

    distance_delta = -10.0 / np.hypot(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
    expected = (
        distance_delta * config.REWARD_CHECKPOINT_APPROACH * config.REWARD_WRONG_WAY_MULTIPLIER
        + config.REWARD_TIME_PENALTY
    )
    assert reward == pytest.approx(expected)
    assert not done
    assert info["forward_progress"] == pytest.approx(0.0)
