"""Car-physics invariants for realistic handling."""
import numpy as np

from game.car import Car
from utils.config import config


def test_car_does_not_turn_in_place():
    car = Car(100.0, 100.0, angle=0.0)
    initial_angle = car.angle

    for _ in range(10):
        car.update(throttle=0.0, steering=1.0, dt=config.DT)

    assert np.isclose(car.angle, initial_angle)
    assert np.isclose(np.linalg.norm(car.velocity), 0.0)


def test_car_keeps_momentum_then_slows_from_friction():
    car = Car(100.0, 100.0, angle=0.0)

    for _ in range(20):
        car.update(throttle=1.0, steering=0.0, dt=config.DT)

    speed_after_accel = np.linalg.norm(car.velocity)
    car.update(throttle=0.0, steering=0.0, dt=config.DT)
    speed_after_coast = np.linalg.norm(car.velocity)

    assert speed_after_accel > 0.0
    assert speed_after_coast > 0.0
    assert speed_after_coast < speed_after_accel


def test_car_turn_rate_depends_on_forward_motion():
    car = Car(100.0, 100.0, angle=0.0)
    angle_before = car.angle

    for _ in range(20):
        car.update(throttle=1.0, steering=0.0, dt=config.DT)

    for _ in range(10):
        car.update(throttle=0.2, steering=1.0, dt=config.DT)

    assert abs(car.angle - angle_before) > 1e-3
