#!/usr/bin/env python3
"""
Training script for the RL racing agent.

This script trains a DQN agent to drive on the procedurally generated tracks.
Training progress is logged and models are periodically saved.

Usage:
    python train.py              # Train with default settings
    python train.py --render    # Train with visualization
    python train.py --episodes 1000  # Train for 1000 episodes
    python train.py --load model.pth  # Continue training from saved model
"""

import argparse
import os
import random
import time
import json
import csv
from datetime import datetime
import numpy as np
import torch

from game.track import Track
from rl.agent import DQNAgent, RandomAgent
from rl.environment import RacingEnv, MultiTrackEnv
from jepa.agent import JEPAAgent
from utils.config import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL agent for racing game')
    
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the training (slower but visual)')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to load model from')
    parser.add_argument('--save-dir', type=str, default='saved_models',
                        help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save training logs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=config.GAMMA,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=config.EPSILON_START,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=config.EPSILON_MIN,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=config.EPSILON_DECAY,
                        help='Exploration decay rate')
    parser.add_argument('--learning-starts', type=int, default=config.LEARNING_STARTS,
                        help='Environment steps to collect before training updates')
    parser.add_argument('--target-update-mode', choices=['polyak', 'hard'],
                        default=config.TARGET_UPDATE_MODE,
                        help='Target network update mode')
    parser.add_argument('--target-update-freq', type=int, default=config.TARGET_UPDATE_FREQ,
                        help='Train steps between hard target updates')
    parser.add_argument('--polyak-tau', type=float, default=config.POLYAK_TAU,
                        help='Soft target update rate for polyak mode')
    parser.add_argument('--prioritized', action='store_true',
                        help='Use prioritized experience replay')
    parser.add_argument('--multi-track', action='store_true',
                        help='Train on multiple tracks')
    parser.add_argument('--num-tracks', type=int, default=3,
                        help='Number of tracks for multi-track training')
    parser.add_argument('--render-every-n', type=int, default=1,
                        help='Render every Nth step when rendering')
    parser.add_argument('--dueling', action='store_true',
                        help='Use Dueling DQN')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED,
                        help='Random seed for reproducible runs')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic torch algorithms where available')
    parser.add_argument('--best-window', type=int, default=50,
                        help='Moving reward window for best-model selection')
    parser.add_argument('--save-freq', type=int, default=config.SAVE_FREQ,
                        help='Save model every N episodes')
    parser.add_argument('--log-freq', type=int, default=config.LOG_FREQ,
                        help='Log training every N episodes')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (run trained agent)')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--approach', type=str, default=config.DEFAULT_APPROACH,
                        choices=['dqn', 'jepa'],
                        help='Learning approach: dqn (reward-based RL) or jepa (self-supervised world model)')
    
    return parser.parse_args()


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy, and torch random sources."""
    # Seed every RNG touched by training so repeated runs differ less by luck than by code.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # CUDA kernels have separate RNG state, so GPU training needs its own seed too.
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingLogger:
    """Logger for training progress."""
    
    def __init__(self, log_dir='logs', save_dir='saved_models'):
        """
        Initialize the logger.
        
        Args:
            log_dir: str - directory for log files
            save_dir: str - directory for saved models
        """
        self.log_dir = log_dir
        self.save_dir = save_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Training timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Log files
        self.csv_path = os.path.join(log_dir, f'training_{self.timestamp}.csv')
        self.json_path = os.path.join(log_dir, f'config_{self.timestamp}.json')
        self.best_model_path = os.path.join(save_dir, f'best_model_{self.timestamp}.pth')
        self.latest_model_path = os.path.join(save_dir, f'model_{self.timestamp}.pth')
        
        # CSV captures episode-by-episode time series; JSON freezes config for later reproduction.
        # Reading both together helps explain not only what happened, but under which setup.
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'avg_reward', 'lap_time', 
                'best_lap_time', 'epsilon', 'loss', 'steps', 'memory_size'
            ])
    
    def log_episode(self, episode, total_reward, avg_reward, lap_time, 
                    best_lap_time, epsilon, loss, steps, memory_size):
        """Log an episode to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, avg_reward, lap_time,
                best_lap_time, epsilon, loss, steps, memory_size
            ])
    
    def save_config(self, args):
        """Save training configuration."""
        config_dict = {
            'timestamp': self.timestamp,
            'args': vars(args),
            'config': {
                'SCREEN_WIDTH': config.SCREEN_WIDTH,
                'SCREEN_HEIGHT': config.SCREEN_HEIGHT,
                'TRACK_WIDTH': config.TRACK_WIDTH,
                'TRACK_COMPLEXITY': config.TRACK_COMPLEXITY,
                'CAR_MAX_SPEED': config.CAR_MAX_SPEED,
                'STATE_DIM': config.STATE_DIM,
                'STATE_VERSION': config.STATE_VERSION,
                'ACTION_DIM': config.ACTION_DIM,
                'NUM_SENSORS': config.NUM_SENSORS,
                'LEARNING_STARTS': config.LEARNING_STARTS,
                'TARGET_UPDATE_MODE': config.TARGET_UPDATE_MODE,
                'TARGET_UPDATE_FREQ': config.TARGET_UPDATE_FREQ,
                'POLYAK_TAU': config.POLYAK_TAU,
            }
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_model(self, agent, is_best=False):
        """Save agent model."""
        if is_best:
            agent.save(self.best_model_path)
        else:
            agent.save(self.latest_model_path)
    
    def print_summary(self, episode, total_reward, avg_reward, lap_time,
                      best_lap_time, epsilon, loss, steps):
        """Print training summary for an episode."""
        print(f"\rEpisode {episode:5d} | "
              f"Reward: {total_reward:8.2f} | "
              f"Avg: {avg_reward:8.2f} | "
              f"Lap: {lap_time:.2f}s | "
              f"Best: {best_lap_time:.2f}s | "
              f"Eps: {epsilon:.4f} | "
              f"Loss: {loss:.4f} | "
              f"Steps: {steps:5d}",
              end='', flush=True)
    
    def print_final_summary(self, total_episodes, total_time, best_lap_time):
        """Print final training summary."""
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Total episodes: {total_episodes}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best lap time: {best_lap_time:.2f} seconds")
        print(f"Logs saved to: {self.log_dir}")
        print(f"Models saved to: {self.save_dir}")
        print("=" * 80)


def train_agent(args):
    """
    Train the RL agent.
    
    Args:
        args: argparse.Namespace - command line arguments
    """
    print("Starting training...")
    print(f"Approach: {args.approach.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print(f"Multi-track: {args.multi_track}")
    if args.approach == 'dqn':
        print(f"Dueling DQN: {args.dueling}")
        print(f"Prioritized replay: {args.prioritized}")
    print()
    
    # Logger gives training artifacts same role as lab notes: rewards over time plus config snapshot.
    logger = TrainingLogger(args.log_dir, args.save_dir)
    logger.save_config(args)
    
    # Create environment
    if args.multi_track:
        env = MultiTrackEnv(
            num_tracks=args.num_tracks,
            render=args.render,
            render_every_n=args.render_every_n,
            seed=args.seed,
        )
    else:
        track = Track(seed=args.seed) if args.seed is not None else None
        env = RacingEnv(track=track, render=args.render, render_every_n=args.render_every_n)
    
    # Create agent based on approach
    if args.approach == 'jepa':
        agent = JEPAAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
        )
        print("Using JEPA (self-supervised world model + CEM planning)")
        print(f"  Warmup steps: {config.JEPA_WARMUP_STEPS} (random exploration)")
        print(f"  Planning horizon: {config.JEPA_PLANNING_HORIZON}")
        print(f"  Latent dim: {config.JEPA_LATENT_DIM}")
        print()
    else:
        agent = DQNAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            target_update_mode=args.target_update_mode,
            target_update_freq=args.target_update_freq,
            polyak_tau=args.polyak_tau,
            use_dueling=args.dueling,
            use_double_dqn=True,
            use_prioritized=args.prioritized,
        )
    
    # Load model if specified
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}...")
        agent.load(args.load)
        print("Model loaded!")
    
    # RL training is episode-based: reset track, then learn from many env steps inside episode.
    start_time = time.time()
    best_lap_time = float('inf')
    best_eval_score = -float('inf')
    running_rewards = []
    best_window = max(1, args.best_window)
    
    try:
        for episode in range(1, args.episodes + 1):
            # Each episode starts fresh so agent sees full rollout from initial state to terminal state.
            state = env.reset()
            
            # Track episode statistics
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            lap_completed = False
            current_lap_time = 0
            
            # Inner loop is interaction phase: act, observe reward, store transition, update policy.
            done = False
            while not done:
                # Select action
                action = agent.select_action(state, explore=True)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Remember transition
                agent.remember(state, action, reward, next_state, done)
                
                # Update agent
                loss = agent.update()
                if loss is not None:
                    episode_loss += loss
                # Exploration schedule follows env steps, because action count drives data collection pace.
                agent.on_env_step()
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Check for lap completion
                if info.get('lap_completed', False):
                    lap_completed = True
                    current_lap_time = info.get('lap_time', 0)
                    
                    if current_lap_time < best_lap_time:
                        best_lap_time = current_lap_time
                 
                # Render
                if args.render:
                    env.render()
            
            # Update agent statistics
            agent.increment_episode()
            
            # Moving average smooths noisy episodic rewards before "best model" decision is made.
            running_rewards.append(episode_reward)
            if len(running_rewards) > best_window:
                running_rewards.pop(0)
            avg_reward = np.mean(running_rewards) if running_rewards else 0
            if avg_reward > best_eval_score:
                best_eval_score = avg_reward
                logger.save_model(agent, is_best=True)
            
            # Get agent stats
            agent_stats = agent.get_stats()
            
            # Log episode
            logger.log_episode(
                episode=episode,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                lap_time=current_lap_time if lap_completed else 0,
                best_lap_time=best_lap_time,
                epsilon=agent.epsilon,
                loss=episode_loss / max(1, episode_steps),
                steps=agent_stats['env_steps'],
                memory_size=len(agent.memory)
            )
            
            # Print summary
            if episode % args.log_freq == 0 or episode == 1:
                logger.print_summary(
                    episode=episode,
                    total_reward=episode_reward,
                    avg_reward=avg_reward,
                    lap_time=current_lap_time if lap_completed else 0,
                    best_lap_time=best_lap_time,
                    epsilon=agent.epsilon,
                    loss=episode_loss / max(1, episode_steps),
                    steps=agent_stats['env_steps']
                )
            
            # Keep latest checkpoint for resume/debug, even when it is worse than best smoothed policy.
            if episode % args.save_freq == 0:
                logger.save_model(agent, is_best=False)
        
        # Final save preserves most recent training state; best save preserves strongest observed policy.
        logger.save_model(agent, is_best=False)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Clean up
        env.close()
        
        # Print final summary
        total_time = time.time() - start_time
        logger.print_final_summary(
            total_episodes=episode,
            total_time=total_time,
            best_lap_time=best_lap_time
        )
    
    return best_lap_time


def test_agent(args):
    """
    Test a trained agent.
    
    Args:
        args: argparse.Namespace - command line arguments
    """
    print(f"\nTesting agent...")
    print(f"Test episodes: {args.test_episodes}")
    print(f"Render: {args.render}")
    
    # Create environment
    track = Track(seed=args.seed) if args.seed is not None else None
    env = RacingEnv(track=track, render=args.render, render_every_n=args.render_every_n)
    
    # Create and load agent
    if args.approach == 'jepa':
        agent = JEPAAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
        )
    else:
        agent = DQNAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=0.0,  # No exploration during testing
            batch_size=args.batch_size
        )
    
    # Load model
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}...")
        agent.load(args.load)
        print("Model loaded!")
    else:
        print("No model loaded. Testing with untrained agent.")
    
    # Evaluation loop measures learned behavior only, so action sampling stays greedy.
    total_rewards = []
    lap_times = []
    completed_laps = 0
    
    for episode in range(args.test_episodes):
        state = env.reset()
        episode_reward = 0
        episode_lap_time = 0
        lap_completed = False
        
        done = False
        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, explore=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            if info.get('lap_completed', False):
                episode_lap_time = info.get('lap_time', 0)
                lap_completed = True
                lap_times.append(episode_lap_time)
                completed_laps += 1
            
            state = next_state
            
            # Render
            if args.render:
                env.render()
        
        total_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Lap Time={'{:.2f}s'.format(episode_lap_time) if lap_completed else 'N/A'}")
    
    # Print summary
    env.close()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"Total episodes: {args.test_episodes}")
    print(f"Completed laps: {completed_laps}/{args.test_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average lap time: {np.mean(lap_times):.2f}s" if lap_times else "No laps completed")
    if lap_times:
        print(f"Best lap time: {np.min(lap_times):.2f}s")
        print(f"Worst lap time: {np.max(lap_times):.2f}s")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed, deterministic=args.deterministic)
    
    if args.test:
        test_agent(args)
    else:
        train_agent(args)


if __name__ == "__main__":
    main()
