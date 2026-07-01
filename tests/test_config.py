"""Config invariants.

These guard the hyperparameter relationships that the rest of the code assumes.
The +4 in STATE_DIM is the classic gotcha (speed + sin + cos + progress on top
of the raw sensor readings), so it gets an explicit test.
"""
from utils.config import config


def test_state_dim_is_sensors_plus_four():
    assert config.STATE_DIM == config.NUM_SENSORS + 4
    assert config.STATE_DIM == 11


def test_action_dim_matches_action_table():
    assert config.ACTION_DIM == len(config.ACTIONS)
    assert set(config.ACTIONS.keys()) == set(range(config.ACTION_DIM))


def test_sensor_angle_count_matches_num_sensors():
    assert len(config.SENSOR_ANGLES) == config.NUM_SENSORS


def test_rl_hyperparameters_in_valid_ranges():
    assert 0 < config.GAMMA <= 1
    assert 0 <= config.EPSILON_MIN <= config.EPSILON_START <= 1
    assert 0 < config.EPSILON_DECAY <= 1
    assert config.LEARNING_RATE > 0
    assert config.BATCH_SIZE > 0
    assert config.MEMORY_SIZE >= config.BATCH_SIZE


def test_default_epsilon_decay_keeps_exploration_after_warmup():
    epsilon_after_1000_training_steps = max(
        config.EPSILON_MIN,
        config.EPSILON_START * (config.EPSILON_DECAY ** 1000),
    )
    epsilon_after_250k_training_steps = max(
        config.EPSILON_MIN,
        config.EPSILON_START * (config.EPSILON_DECAY ** 250_000),
    )
    epsilon_after_1m_training_steps = max(
        config.EPSILON_MIN,
        config.EPSILON_START * (config.EPSILON_DECAY ** 1_000_000),
    )
    assert epsilon_after_1000_training_steps > 0.99
    assert epsilon_after_250k_training_steps > 0.25
    assert epsilon_after_1m_training_steps == config.EPSILON_MIN


def test_stall_reward_settings_make_waiting_worse_than_crashing():
    stall_return = (
        config.REWARD_NO_PROGRESS
        + (config.NO_PROGRESS_PATIENCE_STEPS * config.REWARD_TIME_PENALTY)
    )
    assert stall_return < config.REWARD_COLLISION


def test_every_action_is_bounded():
    for spec in config.ACTIONS.values():
        assert 0.0 <= spec["throttle"] <= 1.0
        assert -1.0 <= spec["steering"] <= 1.0
