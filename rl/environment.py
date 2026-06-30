"""
RL Environment wrapper for the racing game.
Provides a Gym-like interface for reinforcement learning.
"""
import numpy as np
from game.game import Game
from game.track import Track
from utils.config import config


class RacingEnv:
    """
    Reinforcement learning environment for the racing game.
    
    Provides a standard interface similar to OpenAI Gym for RL training.
    Returns the normalized state produced by Game and tracks episode statistics.
    """
    
    def __init__(self, track=None, render=False, max_steps=config.MAX_STEPS_PER_EPISODE,
                 render_every_n=1):
        """
        Initialize the racing environment.
        
        Args:
            track: Track - custom track (None for random generation)
            render: bool - whether to render the environment
            max_steps: int - maximum steps per episode
            render_every_n: int - render every Nth explicit render call
        """
        self.game = Game(track=track, headless=not render, render_every_step=False)
        self._render = render
        self.render_every_n = max(1, int(render_every_n))
        self._render_call_count = 0
        self.max_steps = max_steps
        self.episode = 0
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.lap_times = []
        self.best_lap_time = float('inf')
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            numpy array: initial state
        """
        state = self.game.reset()
        self.episode += 1
        return state
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: int or tuple - action to take
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Map action to game action
        if isinstance(action, int):
            # Discrete action
            action_dict = config.ACTIONS.get(action, config.ACTIONS[0])
            throttle = action_dict['throttle']
            steering = action_dict['steering']
        else:
            throttle, steering = action
        
        # Apply action to game
        game_action = (throttle, steering)
        next_state, reward, done, info = self.game.step(game_action)
        
        # Track episode rewards
        info['episode'] = self.episode
        
        # Check for lap completion
        if info.get('lap_completed', False):
            lap_time = info.get('lap_time', 0)
            self.lap_times.append(lap_time)
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
        
        return next_state, reward, done, info
    
    def close(self):
        """Close the environment and clean up resources."""
        self.game.close()
    
    def get_stats(self):
        """
        Get environment statistics.
        
        Returns:
            dict: environment statistics
        """
        return {
            'episode': self.episode,
            'best_lap_time': self.best_lap_time,
            'avg_lap_time': np.mean(self.lap_times) if self.lap_times else 0,
            'lap_count': len(self.lap_times),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: str - render mode ('human', 'rgb_array', etc.)
        """
        if self._render:
            self._render_call_count += 1
            if self._render_call_count % self.render_every_n == 0:
                self.game.render()
    
    def set_track(self, track):
        """
        Set a new track for the environment.
        
        Args:
            track: Track - new track
        """
        self.game.track = track
        if hasattr(track, 'reset_progress_hint'):
            track.reset_progress_hint()
        # Reset car to new track start
        start_pos = track.start_position
        start_angle = track.start_angle
        self.game.car.reset(start_pos, start_angle)


class MultiTrackEnv:
    """
    Environment that cycles through multiple tracks.
    
    Useful for training on a variety of tracks to improve generalization.
    """
    
    def __init__(self, num_tracks=5, render=False, max_steps=config.MAX_STEPS_PER_EPISODE, 
                 rotation_every=10, eval_mode=False, render_every_n=1, seed=None):
        """
        Initialize the multi-track environment.
        
        Args:
            num_tracks: int - number of tracks to generate
            render: bool - whether to render
            max_steps: int - maximum steps per episode
            rotation_every: int - rotate track every N episodes (default: 10)
            eval_mode: bool - if True, rotate track every reset (for testing)
            render_every_n: int - render every Nth explicit render call
            seed: int - optional base seed for reproducible track generation
        """
        self.num_tracks = num_tracks
        self.tracks = []
        self.rotation_every = 1 if eval_mode else rotation_every
        self.eval_mode = eval_mode
        self.seed = seed
        
        # Generate multiple tracks
        for i in range(num_tracks):
            track_seed = None if seed is None else seed + i
            track = Track(complexity=config.TRACK_COMPLEXITY + i * 2, seed=track_seed)
            self.tracks.append(track)
        
        # Create environment with first track
        self.env = RacingEnv(
            self.tracks[0],
            render=render,
            max_steps=max_steps,
            render_every_n=render_every_n,
        )
        self.current_track_idx = 0
    
    def reset(self):
        """
        Reset the environment, possibly changing to a new track.
        
        Returns:
            numpy array: initial state
        """
        # Change track periodically
        if self.env.episode > 0 and self.env.episode % self.rotation_every == 0:
            self.current_track_idx = (self.current_track_idx + 1) % self.num_tracks
            self.env.set_track(self.tracks[self.current_track_idx])
        
        return self.env.reset()
    
    def step(self, action):
        """
        Execute one step.
        
        Args:
            action: action to take
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        return self.env.step(action)
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_stats(self):
        """Get environment statistics."""
        return self.env.get_stats()


if __name__ == "__main__":
    # Test the environment
    print("Testing RacingEnv...")
    
    env = RacingEnv(render=False)
    
    for episode in range(3):
        state = env.reset()
        print(f"Episode {episode + 1}")
        print(f"  Initial state shape: {state.shape}")
        print(f"  Initial state: {state[:5]}...")  # First 5 values
        
        total_reward = 0
        for step in range(100):
            # Random action
            action = np.random.randint(0, config.ACTION_DIM)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            
            if done:
                break
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {step + 1}")
        print(f"  Progress: {info.get('progress', 0) * 100:.1f}%")
    
    env.close()
    print("RacingEnv test passed!")
