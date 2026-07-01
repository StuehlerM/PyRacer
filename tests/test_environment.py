"""Environment contract: reset/step return the documented (obs, reward, done, info)
shape, observations are finite, and sensor readings are normalized.

These run headless thanks to the SDL dummy driver set in conftest.py.
"""
import numpy as np
import pytest

from game.environment import RacingEnv
from utils.config import config


@pytest.fixture
def env():
    e = RacingEnv(render=False)
    yield e
    e.close()


def test_reset_returns_correct_state(env):
    state = env.reset()
    assert state.shape == (config.STATE_DIM,)
    assert np.all(np.isfinite(state))


def test_step_returns_obs_reward_done_info(env):
    env.reset()
    obs, reward, done, info = env.step(0)
    assert obs.shape == (config.STATE_DIM,)
    assert np.isfinite(float(reward))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_sensor_readings_are_normalized(env):
    state = env.reset()
    sensors = state[: config.NUM_SENSORS]
    assert np.all(sensors >= -1e-6)
    assert np.all(sensors <= 1.0 + 1e-6)


def test_progress_starts_low_and_is_bounded(env):
    state = env.reset()
    progress = state[-1]
    assert 0.0 - 1e-6 <= progress <= 1.0 + 1e-6


def test_episode_runs_to_termination(env):
    """A full-throttle-straight policy leaves the curved track and crashes,
    proving the episode loop actually closes."""
    env.reset()
    done = False
    for _ in range(600):
        _, _, done, _ = env.step(0)
        if done:
            break
    assert done


def test_discrete_steering_actions_reach_car_physics(env):
    steer_actions = {
        "left": next(
            action_id
            for action_id, action in config.ACTIONS.items()
            if action["steering"] < 0.0
        ),
        "right": next(
            action_id
            for action_id, action in config.ACTIONS.items()
            if action["steering"] > 0.0
        ),
    }

    for action_id, expected_sign in (
        (steer_actions["left"], -1.0),
        (steer_actions["right"], 1.0),
    ):
        env.reset()
        initial_angle = env.game.car.angle

        for _ in range(30):
            _, _, done, _ = env.step(action_id)
            if done:
                break

        angle_delta = (env.game.car.angle - initial_angle + np.pi) % (2 * np.pi) - np.pi
        assert expected_sign * env.game.car.steering_angle > 0.0
        assert expected_sign * angle_delta > 1e-3
