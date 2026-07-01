import pytest

from game.game import Game
from game.track import Track
from utils.config import config


def _make_headless_game(seed=0):
    game = Game(track=Track(seed=seed), headless=True)
    game.reset()
    return game


def test_closing_distance_to_next_checkpoint_is_rewarded(monkeypatch):
    game = _make_headless_game()
    game.prev_checkpoint_distance = 0.2

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(game.track, "get_checkpoint_progress", lambda checkpoint_idx: 0.2)
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (0.1, 0.0) if return_distance else 0.1
        ),
    )

    _, reward, done, info = game.step(0)
    expected = (0.2 - 0.1) * config.REWARD_PROGRESS + config.REWARD_TIME_PENALTY
    assert reward == pytest.approx(expected)
    assert not done
    assert info["off_track"] is False


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


def test_backward_start_line_wrap_does_not_count_as_forward_progress(monkeypatch):
    game = _make_headless_game()
    game.prev_progress = 0.01
    game.prev_checkpoint_distance = 0.19

    monkeypatch.setattr(game.car, "update", lambda throttle, steering, dt: None)
    monkeypatch.setattr(game.car, "check_collision", lambda track: (False, None))
    monkeypatch.setattr(game.track, "check_checkpoint", lambda pos, last_idx: (last_idx, False))
    monkeypatch.setattr(game.track, "get_checkpoint_progress", lambda checkpoint_idx: 0.2)
    monkeypatch.setattr(
        game.track,
        "get_progress",
        lambda pos, last_idx_hint=None, return_idx=False, return_distance=False: (
            (0.99, 0.0) if return_distance else 0.99
        ),
    )

    _, reward, done, info = game.step(0)

    expected = ((0.19 - 0.21) * config.REWARD_PROGRESS * config.REWARD_WRONG_WAY_MULTIPLIER
                + config.REWARD_TIME_PENALTY)
    assert reward == pytest.approx(expected)
    assert not done
    assert info["forward_progress"] == pytest.approx(0.0)
