"""
Main game class that manages the game loop, rendering, and RL interface.
"""
import pygame
import numpy as np
from utils.config import config
from .track import Track
from .car import Car


class Game:
    """
    Main game class that manages the game state and loop.
    
    Supports both human play and RL training modes.
    """
    
    def __init__(self, track=None, headless=False, render_every_step=True, render_every_n=1):
        """
        Initialize the game.
        
        Args:
            track: Track - custom track (default: generates new track)
            headless: bool - run without rendering (for RL training)
            render_every_step: bool - render automatically from step()
            render_every_n: int - render every Nth automatic render step
        """
        self.headless = headless
        self.render_every_step = render_every_step
        self.render_every_n = max(1, int(render_every_n))
        self._render_step_counter = 0
        
        # Initialize pygame if not headless
        if not headless:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption(config.TITLE)
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            self.clock = pygame.time.Clock()
            self._font_hud = pygame.font.SysFont(None, 24)
            self._font_pause = pygame.font.SysFont(None, 72)
        else:
            self.screen = None
            self.clock = None
            self._font_hud = None
            self._font_pause = None
        
        # Create track
        self.track = track or Track()
        
        # Create car
        start_pos = self.track.start_position
        start_angle = self.track.start_angle
        self.car = Car(start_pos[0], start_pos[1], start_angle)
        
        # Game state
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.lap_count = 0
        self.last_checkpoint_idx = -1
        self.lap_time = 0.0
        self.best_lap_time = float('inf')
        self._completed_lap_time = 0.0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        
        # For human controls
        self.human_control = not headless
        
        # Learning mode gates RL-only work; human play does not need sensor/state math every frame.
        self.learning_mode = headless
        
        # For RL
        self.prev_progress = self.track.get_progress(tuple(self.car.position))
        self.steps_in_episode = 0
        self.max_steps = config.MAX_STEPS_PER_EPISODE
    
    def reset(self):
        """
        Reset the game to initial state.
        
        Returns:
            numpy array: initial state observation
        """
        # Reset car to start
        start_pos = self.track.start_position
        start_angle = self.track.start_angle
        self.car.reset(start_pos, start_angle)
        if hasattr(self.track, 'reset_progress_hint'):
            self.track.reset_progress_hint()
        
        # Reset game state
        self.lap_count = 0
        self.last_checkpoint_idx = -1
        self.lap_time = 0.0
        self._completed_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.episode_reward = 0.0
        self.steps_in_episode = 0
        self.prev_progress = self.track.get_progress(tuple(self.car.position))
        self._render_step_counter = 0
        
        # Return initial observation
        return self._get_state()
    
    def step(self, action=None):
        """
        Execute one game step.
        
        Args:
            action: int or tuple - RL action (None for human controls)
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Same `step()` serves keyboard play and RL, following common (obs, reward, done, info) API.
        if action is None and self.human_control and not self.headless:
            action = self._handle_human_input()
        
        # Initialize new_best flag for this step
        new_best = False
        
        # Map action to throttle and steering
        if action is not None:
            if isinstance(action, int):
                # Discrete policy outputs action index; config maps it to continuous controls.
                action_dict = config.ACTIONS.get(action, config.ACTIONS[0])
                throttle = action_dict['throttle']
                steering = action_dict['steering']
            else:
                # Assume it's already (throttle, steering)
                throttle, steering = action
        else:
            throttle, steering = 0.0, 0.0
        
        # Update car
        self.car.update(throttle, steering, config.DT)
        
        # Increment lap time at the top - makes relationship to physics ticks clearer
        self.lap_time += config.DT
        
        # Increment step counter
        self.steps_in_episode += 1
        
        # Cache once per frame because collision/progress are reused by reward, HUD, and next state.
        # Check collision (expensive - iterates through all segments)
        is_colliding, collision_info = self.car.check_collision(self.track)

        # Wall-segment collision only fires when the car is near a boundary. A car that
        # leaves the playfield (e.g. an untrained policy driving straight off the track)
        # never gets close to a wall again, so without this it would run until max_steps
        # with no terminal signal. Treat going out of bounds as a crash.
        if not is_colliding and self._is_out_of_bounds():
            is_colliding = True
            collision_info = {'collision': True, 'reason': 'out_of_bounds'}
        
        # Check checkpoint
        new_checkpoint_idx, is_finish = self.track.check_checkpoint(
            tuple(self.car.position), self.last_checkpoint_idx
        )
        
        # Calculate progress (expensive - iterates through all waypoints)
        current_progress = self.track.get_progress(tuple(self.car.position))
        
        # If in human driving mode (learning_mode=False), skip all learning-related calculations
        if self.learning_mode:
            # Reward shaping pays for forward motion so agent learns before full laps become common.
            raw_diff = current_progress - self.prev_progress
            # If we apparently went backwards by > 0.5 of the track, treat it as a wrap.
            if raw_diff < -0.5:
                raw_diff += 1.0
            elif raw_diff > 0.5:
                # Teleport / glitch — treat as zero progress this step
                raw_diff = 0.0
            progress_reward = raw_diff * config.REWARD_PROGRESS
            self.prev_progress = current_progress
            
            # RL reward is sum of small hints, not only sparse win/lose events.
            reward = 0.0
            reward += progress_reward
            
            # Initialize done flag for this step
            done = False
            
            # Checkpoints give medium-term goals so long tracks do not feel reward-empty.
            if new_checkpoint_idx > self.last_checkpoint_idx:
                self.last_checkpoint_idx = new_checkpoint_idx
                reward += config.REWARD_CHECKPOINT
                
                if is_finish:
                    # Lap completed!
                    current_lap_time = self.lap_time
                    self.lap_count += 1
                    
                    # Best-lap bonus nudges policy toward speed, not only safe completion.
                    if current_lap_time < self.best_lap_time:
                        self.best_lap_time = current_lap_time
                        new_best = True
                        lap_reward = config.REWARD_LAP_COMPLETE * 1.5  # Bonus for new best
                    else:
                        new_best = False
                        lap_reward = config.REWARD_LAP_COMPLETE
                    
                    # Store completed lap time for info dict BEFORE resetting
                    self._completed_lap_time = current_lap_time
                    self.lap_time = 0.0  # Reset AFTER recording
                    
                    # Reset checkpoint tracking for next lap
                    self.last_checkpoint_idx = -1
                    
                    reward += lap_reward
            
            # Crashes end episode so replay buffer clearly labels bad trajectories as terminal.
            if is_colliding:
                reward += config.REWARD_COLLISION
                done = True
            
            # Small living cost discourages idling and forces agent to trade safety vs pace.
            reward += config.REWARD_TIME_PENALTY
            
            # Update total rewards
            self.episode_reward += reward
            self.total_reward += reward
        else:
            # Human mode keeps game rules but skips RL bookkeeping to save sensor/calc cost.
            
            # Initialize done flag for this step
            done = False
            
            if new_checkpoint_idx > self.last_checkpoint_idx:
                self.last_checkpoint_idx = new_checkpoint_idx
                
                if is_finish:
                    # Lap completed!
                    current_lap_time = self.lap_time
                    self.lap_count += 1
                    
                    # Update best lap time
                    if current_lap_time < self.best_lap_time:
                        self.best_lap_time = current_lap_time
                        new_best = True
                    else:
                        new_best = False
                    
                    # Store completed lap time for info dict BEFORE resetting
                    self._completed_lap_time = current_lap_time
                    self.lap_time = 0.0  # Reset AFTER recording
                    
                    # Reset checkpoint tracking for next lap
                    self.last_checkpoint_idx = -1
            
            # In human mode, reward stays at 0
            reward = 0.0
            self.prev_progress = current_progress
        
        # Check if episode is done
        if self.steps_in_episode >= self.max_steps:
            done = True
        
        # Handle collision in human mode too
        if is_colliding and not self.learning_mode:
            # In human mode, we still need to detect collisions for display
            # but we don't apply penalties
            pass
        
        # Observation is next state after action, matching standard RL environment contract.
        state = self._get_state(
            sensor_readings=None,  # Will be computed only if needed
            progress=current_progress,  # Use cached progress
            skip_sensors=not self.learning_mode  # Skip expensive sensor computation in human mode
        )
        
        # `info` carries diagnostics for training logs without polluting learning signal.
        info = {
            'lap_count': self.lap_count,
            'lap_time': self._completed_lap_time if is_finish else self.lap_time,
            'best_lap_time': self.best_lap_time,
            'collision': is_colliding,
            'progress': current_progress,
            'lap_completed': is_finish,
            'new_best_lap': is_finish and new_best,
            'episode_reward': self.episode_reward,
            'steps': self.steps_in_episode
        }
        
        # Render if automatic rendering is enabled - pass cached values
        if not self.headless and self.render_every_step:
            self._render_step_counter += 1
            if self._render_step_counter % self.render_every_n == 0:
                self.render(progress=current_progress, is_colliding=is_colliding)
        
        return state, reward, done, info

    def _is_out_of_bounds(self):
        """True if the car left the screen by more than config.OOB_MARGIN pixels."""
        margin = config.OOB_MARGIN
        x, y = self.car.position
        return (
            x < -margin or x > config.SCREEN_WIDTH + margin or
            y < -margin or y > config.SCREEN_HEIGHT + margin
        )
    
    def _get_state(self, sensor_readings=None, progress=None, skip_sensors=False):
        """
        Get the current state observation for RL.
        
        Args:
            sensor_readings: list - pre-computed sensor readings (optional)
            progress: float - pre-computed progress (optional)
            skip_sensors: bool - if True and no sensor_readings provided, use zeros instead of computing
        
        Returns:
            numpy array: state vector
        """
        # Get sensor readings if not provided
        if sensor_readings is None:
            if skip_sensors:
                # Zero-fill keeps state shape stable even when no agent is consuming sensors.
                sensor_readings = [0.0] * config.NUM_SENSORS
            else:
                sensor_readings = self.car.get_sensor_readings(self.track)
        
        # State mixes perception, motion, heading, and track progress into fixed-size RL observation.
        car_speed = self.car.speed / self.car.max_speed  # Normalize speed
        car_angle = self.car.angle
        
        # Use provided progress or calculate
        if progress is None:
            progress = self.track.get_progress(tuple(self.car.position))
        
        # sin/cos avoid angle wrap jump at ±pi, which is easier for neural net than raw radians.
        state = np.concatenate([
            sensor_readings,
            [car_speed, np.sin(car_angle), np.cos(car_angle), progress]
        ])
        
        return state.astype(np.float32)
    
    def _handle_human_input(self):
        """
        Handle human keyboard input.
        Uses ONLY arrow keys for driving to free up letter keys for other functions.
        
        Returns:
            tuple: (throttle, steering)
        """
        throttle = 0.0
        steering = 0.0
        
        keys = pygame.key.get_pressed()
        
        # Arrow-only driving frees letter keys for debug toggles without conflicting with steering.
        if keys[pygame.K_UP]:
            throttle = 1.0
        
        # Brake/Reverse - ONLY arrow down
        if keys[pygame.K_DOWN]:
            throttle = -1.0
        
        # Turn left - ONLY arrow left
        if keys[pygame.K_LEFT]:
            steering = -1.0
        
        # Turn right - ONLY arrow right
        if keys[pygame.K_RIGHT]:
            steering = 1.0
        
        # Reset car (for testing) - uses R key
        if keys[pygame.K_r]:
            self.reset()
        
        return throttle, steering
    
    def render(self, progress=None, is_colliding=False):
        """
        Render the game state to the screen.
        
        Args:
            progress: float - pre-computed progress (optional)
            is_colliding: bool - pre-computed collision state (optional)
        """
        if self.headless or self.screen is None:
            return

        # Pump the OS event queue every frame. Without this the training window
        # (e.g. `train.py --render`) is never serviced, so Windows marks it
        # "Not Responding" and the process can crash. Closing the window stops
        # rendering cleanly instead of raising on the next draw/flip.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.close()
                self.headless = True
                return

        try:
            # Clear screen
            self.screen.fill(config.Colors.BLACK)

            # Draw track
            self.track.draw(self.screen)

            # Draw car
            self.car.draw(self.screen)

            # Optionally draw sensors (press 's' to toggle) - only in learning mode
            keys = pygame.key.get_pressed()
            if keys[pygame.K_s] and self.learning_mode:
                self.car.draw_sensors(self.screen)

            # Draw HUD with cached values
            self._draw_hud(progress=progress, is_colliding=is_colliding)

            # Update display
            pygame.display.flip()
        except pygame.error:
            # Display was closed/lost mid-render; stop rendering rather than crash training.
            self.headless = True
            return

        self.frame_count += 1
    
    def _draw_hud(self, progress=None, is_colliding=False):
        """
        Draw the heads-up display with game information.
        
        Args:
            progress: float - pre-computed progress (optional)
            is_colliding: bool - pre-computed collision state (optional)
        """
        font = self._font_hud
        
        # Mode indicator (top-right)
        mode_text = font.render(f"Mode: {'LEARNING' if self.learning_mode else 'HUMAN'}", 
                               True, config.Colors.CYAN if self.learning_mode else config.Colors.WHITE)
        self.screen.blit(mode_text, (config.SCREEN_WIDTH - 150, 10))
        
        # Speed
        speed_text = font.render(f"Speed: {abs(self.car.speed):.1f}", True, config.Colors.WHITE)
        self.screen.blit(speed_text, (10, 10))
        
        # Lap count
        lap_text = font.render(f"Laps: {self.lap_count}", True, config.Colors.WHITE)
        self.screen.blit(lap_text, (10, 40))
        
        # Lap time
        lap_time_text = font.render(f"Lap Time: {self.lap_time:.2f}s", True, config.Colors.WHITE)
        self.screen.blit(lap_time_text, (10, 70))
        
        # Best lap time
        if self.best_lap_time < float('inf'):
            best_text = font.render(f"Best: {self.best_lap_time:.2f}s", True, config.Colors.GREEN)
            self.screen.blit(best_text, (10, 100))
        
        # Episode reward (for RL)
        reward_text = font.render(f"Reward: {self.episode_reward:.2f}", True, config.Colors.YELLOW)
        self.screen.blit(reward_text, (10, 130))
        
        # Progress - use cached value or calculate if not provided
        if progress is None:
            progress = self.track.get_progress(tuple(self.car.position))
        progress_text = font.render(f"Progress: {progress*100:.1f}%", True, config.Colors.WHITE)
        self.screen.blit(progress_text, (10, 160))
        
        # Collision indicator - use cached value or check if not provided
        if is_colliding:
            collision_text = font.render("COLLISION!", True, config.Colors.RED)
            self.screen.blit(collision_text, (config.SCREEN_WIDTH // 2 - 50, 10))
    
    def run(self):
        """
        Run the main game loop for human play.
        """
        if self.headless:
            print("Cannot run human play in headless mode.")
            return
        
        self.running = True
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_l:
                        # Toggle learning mode
                        self.learning_mode = not self.learning_mode
                        mode_name = "Learning" if self.learning_mode else "Human"
                        print(f"Switched to {mode_name} mode")
            
            if self.paused:
                # Draw paused message
                text = self._font_pause.render("PAUSED", True, config.Colors.WHITE)
                text_rect = text.get_rect(center=(config.SCREEN_WIDTH//2, config.SCREEN_HEIGHT//2))
                self.screen.blit(text, text_rect)
                pygame.display.flip()
                self.clock.tick(config.FPS)
                continue
            
            # Run game step
            _, _, _, _ = self.step(None)
            self.clock.tick(config.FPS)
        
        pygame.quit()
    
    def close(self):
        """
        Clean up game resources.
        """
        if not self.headless and self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # Run the game directly
    game = Game(headless=False)
    game.run()
