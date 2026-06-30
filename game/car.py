"""
Car class for the racing game with physics and rendering.
"""
import numpy as np
import pygame
from utils.config import config
from .physics import rotate_vector, clamp, s_curve_acceleration, log_curve_acceleration


class Car:
    """
    Represents a 2D car with physics for the racing game.
    
    The car has position, velocity, angle, and can accelerate, brake, and steer.
    Collision handling is done separately in the game engine.
    """
    
    def __init__(self, x, y, angle=0):
        """
        Initialize a new car.
        
        Args:
            x: float - initial x position
            y: float - initial y position
            angle: float - initial angle in radians (0 = facing right)
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.angle = float(angle)  # In radians
        self.speed = 0.0  # Scalar speed
        self.steering_angle = 0.0  # Current steering angle in radians
        
        # Car dimensions
        self.width = config.CAR_WIDTH
        self.height = config.CAR_HEIGHT
        
        # Physics parameters
        self.max_speed = config.CAR_MAX_SPEED
        self.acceleration = config.CAR_ACCELERATION
        self.braking = config.CAR_BRAKING
        self.friction = config.CAR_FRICTION
        self.steering_speed = np.deg2rad(config.CAR_STEERING_SPEED)
        self.max_steering = np.deg2rad(config.CAR_MAX_STEERING_ANGLE)
        self.steering_return_speed = np.deg2rad(config.CAR_STEERING_RETURN_SPEED)
        
        # For rendering
        self.color = config.Colors.CAR_COLOR
        
        # For sensors (RL agent)
        self.sensor_readings = [0.0] * config.NUM_SENSORS
        self._cached_nearby_segments = None
        self._cached_nearby_position = None
    
    def update(self, throttle=0.0, steering=0.0, dt=config.DT):
        """
        Update the car's physics based on input controls.
        
        Args:
            throttle: float - -1.0 to 1.0 (negative = brake/reverse, positive = accelerate)
            steering: float - -1.0 to 1.0 (negative = left, positive = right)
            dt: float - time delta (default: 1/60)
        """
        # Apply steering
        steering_input = steering * self.steering_speed * dt
        self.steering_angle = clamp(self.steering_angle + steering_input, 
                                     -self.max_steering, self.max_steering)
        
        # Real cars self-center; easing wheel back prevents "stuck turn" feel in digital input.
        if abs(steering) < 1e-3:  # No steering input
            # Return to centre at most `steering_return_speed * dt` per tick.
            max_step  = self.steering_return_speed * dt
            step_size = min(max_step, abs(self.steering_angle))
            self.steering_angle -= np.sign(self.steering_angle) * step_size
            if abs(self.steering_angle) < 1e-4:
                self.steering_angle = 0.0
        
        # Apply throttle/brake with configurable acceleration curve
        if throttle > 0:
            # Nonlinear curves fake engine/drag behavior without simulating full drivetrain physics.
            curve_type = config.ACCELERATION_CURVE
            if curve_type == "s_curve":
                acceleration_amount = s_curve_acceleration(
                    self.speed, self.max_speed, self.acceleration, throttle, dt
                )
            elif curve_type == "log_curve":
                acceleration_amount = log_curve_acceleration(
                    self.speed, self.max_speed, self.acceleration, throttle, dt
                )
            else:  # "linear" or default
                acceleration_amount = throttle * self.acceleration * dt
            self.speed += acceleration_amount
        elif throttle < 0:
            # Brake or reverse
            self.speed += throttle * self.braking * dt
        
        # Apply friction
        if abs(self.speed) > 0:
            self.speed -= np.sign(self.speed) * self.friction * dt
            if abs(self.speed) < 0.001:
                self.speed = 0
        
        # Clamp speed
        self.speed = clamp(self.speed, -self.max_speed, self.max_speed)
        
        # Steering needs forward motion; below threshold we avoid sideways-looking pivot behavior.
        if abs(self.speed) < 1.0:
            # Just move in the current direction
            direction = np.array([np.cos(self.angle), np.sin(self.angle)])
            self.velocity = direction * self.speed
        else:
            # `effective_angle` is where wheels want velocity to go, not raw body heading alone.
            effective_angle = self.angle + self.steering_angle
            direction = np.array([np.cos(effective_angle), np.sin(effective_angle)])
            self.velocity = direction * self.speed
        
        # Update position
        self.position += self.velocity * dt
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
        
        if hasattr(self, '_cached_nearby_position') and self._cached_nearby_position is not None:
            # Nearby-wall cache only matters locally; refresh after big move to avoid stale segments.
            if np.linalg.norm(self.position - self._cached_nearby_position) > 50:
                self._cached_nearby_segments = None
                self._cached_nearby_position = None
    
    def get_state(self):
        """
        Get the car's state as a dictionary.
        
        Returns:
            dict: car state with position, velocity, angle, speed
        """
        return {
            'position': tuple(self.position),
            'velocity': tuple(self.velocity),
            'angle': self.angle,
            'speed': self.speed,
            'steering_angle': self.steering_angle
        }
    
    def get_sensor_readings(self, track):
        """
        Get sensor readings from the car's sensors.
        
        Uses raycasting to determine distances to track boundaries.
        
        Args:
            track: Track - the track to sense
        
        Returns:
            list: normalized sensor readings (0-1)
        """
        from .physics import ray_cast
        
        readings = []
        max_distance = config.SENSOR_MAX_DISTANCE
        
        # Track can expose NumPy-ready arrays so sensors skip rebuilding segment lists every frame.
        if hasattr(track, 'get_boundary_arrays'):
            boundaries = track.get_boundary_arrays()
        else:
            boundaries = track.get_boundaries()
        
        # Reusing car sin/cos avoids 7 extra trig calls per frame, which matters during training.
        car_cos = np.cos(self.angle)
        car_sin = np.sin(self.angle)
        
        # Spatial partitioning narrows ray tests to nearby walls instead of whole track.
        nearby_segment_indices = None
        if hasattr(track, 'get_nearby_segments'):
            # Sensors share same origin, so one nearby lookup can serve every ray this frame.
            if not hasattr(self, '_cached_nearby_segments') or self._cached_nearby_segments is None:
                self._cached_nearby_segments = track.get_nearby_segments(
                    tuple(self.position), max_distance + 50
                )
                self._cached_nearby_position = self.position.copy()
            nearby_segment_indices = self._cached_nearby_segments
        
        for i, sensor_angle in enumerate(config.SENSOR_ANGLES):
            # Absolute sensor angle = car angle + sensor angle
            absolute_angle = self.angle + sensor_angle
            
            # Angle-sum identities rotate each sensor from car heading cheaper than fresh trig pairs.
            # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            sensor_cos = np.cos(sensor_angle)
            sensor_sin = np.sin(sensor_angle)
            direction_x = car_cos * sensor_cos - car_sin * sensor_sin
            direction_y = car_sin * sensor_cos + car_cos * sensor_sin
            direction = np.array([direction_x, direction_y])
            
            # Raycast from car position in sensor direction
            distance = ray_cast(
                self.position, 
                direction, 
                max_distance, 
                boundaries,
                nearby_segment_indices
            )
            
            # RL agent learns from scale-consistent inputs; 0 means wall on bumper, 1 means clear lane.
            normalized = distance / max_distance
            if normalized > 1.0:
                normalized = 1.0
            readings.append(normalized)
        
        self.sensor_readings = readings
        return readings
    
    def check_collision(self, track):
        """
        Check if the car is colliding with the track.
        
        Args:
            track: Track - the track to check against
        
        Returns:
            tuple: (is_colliding, collision_info)
        """
        # Circle is cheaper than rotated-box collision and good enough for forgiving arcade handling.
        radius = np.sqrt(self.width**2 + self.height**2) / 2
        
        return track.check_collision(tuple(self.position), radius)
    
    def reset(self, position, angle=0):
        """
        Reset the car to a new position and angle.
        
        Args:
            position: tuple (x, y) - new position
            angle: float - new angle in radians
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.angle = (float(angle) + np.pi) % (2 * np.pi) - np.pi
        self.speed = 0.0
        self.steering_angle = 0.0
        self._cached_nearby_segments = None
        self._cached_nearby_position = None
    
    def draw(self, screen):
        """
        Draw the car on a pygame screen.
        
        Args:
            screen: pygame.Surface - surface to draw on
        """
        # Calculate car corners
        # Car is drawn as a rectangle oriented at self.angle
        
        # Half dimensions
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Four corners relative to center
        corners = [
            (-half_width, -half_height),  # Bottom-left
            (half_width, -half_height),   # Bottom-right
            (half_width, half_height),     # Top-right
            (-half_width, half_height)     # Top-left
        ]
        
        # Rotation turns local car shape into world-space polygon for drawing.
        rotated_corners = []
        for x, y in corners:
            # Rotate
            rotated_x = x * np.cos(self.angle) - y * np.sin(self.angle)
            rotated_y = x * np.sin(self.angle) + y * np.cos(self.angle)
            
            # Translate to car position
            screen_x = self.position[0] + rotated_x
            screen_y = self.position[1] + rotated_y
            
            rotated_corners.append((screen_x, screen_y))
        
        # Draw car body
        pygame.draw.polygon(screen, self.color, rotated_corners)
        
        # Draw car outline
        pygame.draw.polygon(screen, config.Colors.WHITE, rotated_corners, 2)
        
        # Front marker makes heading obvious because rectangle alone is symmetric at speed.
        front_center = (
            self.position[0] + half_height * np.cos(self.angle),
            self.position[1] + half_height * np.sin(self.angle)
        )
        pygame.draw.circle(screen, config.Colors.RED, (int(front_center[0]), int(front_center[1])), 5)
        
        # Draw steering indicator
        if abs(self.steering_angle) > 0.01:
            steer_dir = np.sign(self.steering_angle)
            steer_x = self.position[0] + 30 * np.cos(self.angle + steer_dir * np.pi/4)
            steer_y = self.position[1] + 30 * np.sin(self.angle + steer_dir * np.pi/4)
            pygame.draw.line(screen, config.Colors.YELLOW, 
                           self.position, (steer_x, steer_y), 2)
    
    def draw_sensors(self, screen):
        """
        Draw the car's sensors (for debugging).
        
        Args:
            screen: pygame.Surface - surface to draw on
        """
        for i, angle in enumerate(config.SENSOR_ANGLES):
            absolute_angle = self.angle + angle
            direction = np.array([np.cos(absolute_angle), np.sin(absolute_angle)])
            
            # Debug rays always draw full length; color encodes actual normalized hit distance.
            end_x = self.position[0] + direction[0] * config.SENSOR_MAX_DISTANCE
            end_y = self.position[1] + direction[1] * config.SENSOR_MAX_DISTANCE
            
            # Draw sensor line
            alpha = int(255 * (1 - self.sensor_readings[i]))
            color = (255, alpha, 0)  # Orange to yellow based on distance
            pygame.draw.line(screen, color, self.position, (end_x, end_y), 1)
            
            # Draw sensor end point
            pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)
