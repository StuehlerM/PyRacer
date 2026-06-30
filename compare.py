#!/usr/bin/env python3
"""
Comparison script: DQN (reward-based RL) vs JEPA (self-supervised world model).

Trains both approaches on the same track and generates comparison plots
showing learning curves, lap completion rates, and behavior differences.

Usage:
    python compare.py                          # Default comparison (500 episodes each)
    python compare.py --episodes 1000          # Longer training
    python compare.py --render                 # Visualize training
    python compare.py --output comparison_results  # Custom output dir

What you'll see:
    - DQN learns from REWARD signal (trial & error → reward → update Q-values)
    - JEPA learns from PREDICTION (observe transitions → build world model → plan)
    - DQN typically learns faster on simple tasks (direct reward gradient)
    - JEPA learns more general world understanding (transferable)
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
from rl.agent import DQNAgent
from rl.environment import RacingEnv
from jepa.agent import JEPAAgent
from utils.config import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare DQN vs JEPA approaches')
    
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes per approach')
    parser.add_argument('--render', action='store_true',
                        help='Render training (slower)')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='Output directory for results and plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Log every N episodes')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate matplotlib comparison plots')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    
    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingMetrics:
    """Collects metrics during training for comparison."""
    
    def __init__(self, name):
        self.name = name
        self.episode_rewards = []
        self.episode_progress = []
        self.episode_steps = []
        self.lap_completions = []
        self.lap_times = []
        self.losses = []
        self.epsilons = []
        self.timestamps = []  # Wall-clock time
    
    def log(self, reward, progress, steps, lap_completed, lap_time, loss, epsilon):
        """Log metrics for one episode."""
        self.episode_rewards.append(reward)
        self.episode_progress.append(progress)
        self.episode_steps.append(steps)
        self.lap_completions.append(1 if lap_completed else 0)
        self.lap_times.append(lap_time if lap_completed else 0)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.timestamps.append(time.time())
    
    def get_rolling_mean(self, values, window=50):
        """Compute rolling mean for smoother plots."""
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(np.mean(values[start:i+1]))
        return result
    
    def summary(self):
        """Get summary statistics."""
        return {
            'name': self.name,
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'avg_progress': np.mean(self.episode_progress) if self.episode_progress else 0,
            'lap_completion_rate': np.mean(self.lap_completions) * 100 if self.lap_completions else 0,
            'avg_lap_time': np.mean([t for t in self.lap_times if t > 0]) if any(t > 0 for t in self.lap_times) else 0,
            'best_lap_time': np.min([t for t in self.lap_times if t > 0]) if any(t > 0 for t in self.lap_times) else 0,
            'total_time': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0,
        }


def train_approach(approach, episodes, track_seed, render=False, log_freq=10):
    """
    Train one approach and collect metrics.
    
    Args:
        approach: 'dqn' or 'jepa'
        episodes: Number of training episodes
        track_seed: Seed for track generation (same for both)
        render: Whether to render
        log_freq: Print progress every N episodes
        
    Returns:
        TrainingMetrics with results
    """
    metrics = TrainingMetrics(approach.upper())
    
    # Create environment with same track for fair comparison
    track = Track(seed=track_seed)
    env = RacingEnv(track=track, render=render)
    
    # Create agent
    if approach == 'jepa':
        agent = JEPAAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
        )
    else:
        agent = DQNAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            use_double_dqn=True,
        )
    
    print(f"\n{'='*60}")
    print(f"Training {approach.upper()} for {episodes} episodes")
    print(f"{'='*60}")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0
        lap_completed = False
        lap_time = 0
        max_progress = 0
        
        done = False
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
            agent.on_env_step()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            progress = info.get('progress', 0)
            max_progress = max(max_progress, progress)
            
            if info.get('lap_completed', False):
                lap_completed = True
                lap_time = info.get('lap_time', 0)
            
            if render:
                env.render()
        
        agent.increment_episode()
        
        # Log metrics
        avg_loss = episode_loss / max(1, episode_steps)
        metrics.log(
            reward=episode_reward,
            progress=max_progress,
            steps=episode_steps,
            lap_completed=lap_completed,
            lap_time=lap_time,
            loss=avg_loss,
            epsilon=agent.epsilon,
        )
        
        # Print progress
        if episode % log_freq == 0 or episode == 1:
            print(f"  [{approach.upper()}] Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Progress: {max_progress*100:5.1f}% | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    env.close()
    return metrics


def generate_plots(dqn_metrics, jepa_metrics, output_dir):
    """
    Generate matplotlib comparison plots.
    
    Args:
        dqn_metrics: TrainingMetrics for DQN
        jepa_metrics: TrainingMetrics for JEPA
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot generation.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN (Reward-based RL) vs JEPA (Self-supervised World Model)', fontsize=14)
    
    episodes_dqn = range(1, len(dqn_metrics.episode_rewards) + 1)
    episodes_jepa = range(1, len(jepa_metrics.episode_rewards) + 1)
    
    # --- Plot 1: Episode Rewards ---
    ax = axes[0, 0]
    ax.plot(episodes_dqn, dqn_metrics.get_rolling_mean(dqn_metrics.episode_rewards),
            label='DQN', color='blue', alpha=0.8)
    ax.plot(episodes_jepa, jepa_metrics.get_rolling_mean(jepa_metrics.episode_rewards),
            label='JEPA', color='red', alpha=0.8)
    # Light raw data
    ax.plot(episodes_dqn, dqn_metrics.episode_rewards, color='blue', alpha=0.1)
    ax.plot(episodes_jepa, jepa_metrics.episode_rewards, color='red', alpha=0.1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Reward (rolling avg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Track Progress ---
    ax = axes[0, 1]
    ax.plot(episodes_dqn, dqn_metrics.get_rolling_mean(dqn_metrics.episode_progress),
            label='DQN', color='blue', alpha=0.8)
    ax.plot(episodes_jepa, jepa_metrics.get_rolling_mean(jepa_metrics.episode_progress),
            label='JEPA', color='red', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Progress')
    ax.set_title('Track Progress (rolling avg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # --- Plot 3: Lap Completion Rate ---
    ax = axes[1, 0]
    window = 50
    dqn_lap_rate = dqn_metrics.get_rolling_mean(dqn_metrics.lap_completions, window)
    jepa_lap_rate = jepa_metrics.get_rolling_mean(jepa_metrics.lap_completions, window)
    ax.plot(episodes_dqn, dqn_lap_rate, label='DQN', color='blue', alpha=0.8)
    ax.plot(episodes_jepa, jepa_lap_rate, label='JEPA', color='red', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Completion Rate')
    ax.set_title(f'Lap Completion Rate (rolling {window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # --- Plot 4: Training Loss ---
    ax = axes[1, 1]
    ax.plot(episodes_dqn, dqn_metrics.get_rolling_mean(dqn_metrics.losses),
            label='DQN (TD loss)', color='blue', alpha=0.8)
    ax.plot(episodes_jepa, jepa_metrics.get_rolling_mean(jepa_metrics.losses),
            label='JEPA (VICReg loss)', color='red', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (rolling avg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")


def save_results(dqn_metrics, jepa_metrics, output_dir):
    """Save comparison results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV with per-episode data
    csv_path = os.path.join(output_dir, f'comparison_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['approach', 'episode', 'reward', 'progress', 'steps',
                        'lap_completed', 'lap_time', 'loss', 'epsilon'])
        
        for i, m in enumerate([dqn_metrics, jepa_metrics]):
            for ep in range(len(m.episode_rewards)):
                writer.writerow([
                    m.name, ep + 1, m.episode_rewards[ep], m.episode_progress[ep],
                    m.episode_steps[ep], m.lap_completions[ep], m.lap_times[ep],
                    m.losses[ep], m.epsilons[ep]
                ])
    
    # Save summary JSON
    json_path = os.path.join(output_dir, f'comparison_summary_{timestamp}.json')
    summary = {
        'timestamp': timestamp,
        'dqn': dqn_metrics.summary(),
        'jepa': jepa_metrics.summary(),
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
    return summary


def print_comparison(dqn_summary, jepa_summary):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("  COMPARISON: DQN (Reward-based RL) vs JEPA (Self-supervised)")
    print("=" * 70)
    print(f"{'Metric':<25} {'DQN':>15} {'JEPA':>15} {'Difference':>15}")
    print("-" * 70)
    
    metrics = [
        ('Avg Reward', 'avg_reward', '.2f'),
        ('Max Reward', 'max_reward', '.2f'),
        ('Avg Progress', 'avg_progress', '.3f'),
        ('Lap Completion %', 'lap_completion_rate', '.1f'),
        ('Avg Lap Time (s)', 'avg_lap_time', '.2f'),
        ('Best Lap Time (s)', 'best_lap_time', '.2f'),
        ('Training Time (s)', 'total_time', '.1f'),
    ]
    
    for label, key, fmt in metrics:
        dqn_val = dqn_summary[key]
        jepa_val = jepa_summary[key]
        diff = jepa_val - dqn_val
        print(f"{label:<25} {dqn_val:>15{fmt}} {jepa_val:>15{fmt}} {diff:>+15{fmt}}")
    
    print("=" * 70)
    print("\nKey observations:")
    print("  • DQN learns from explicit reward signal (faster for simple tasks)")
    print("  • JEPA learns world dynamics without reward (more generalizable)")
    print("  • JEPA needs warmup period for world model before planning works")
    print("  • DQN exploration is ε-greedy; JEPA uses CEM planning after warmup")
    print("=" * 70)


def main():
    """Main comparison entry point."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 70)
    print("  PyRacer: DQN vs JEPA Comparison")
    print("  RL (reward-based) vs Self-supervised (world model + planning)")
    print("=" * 70)
    print(f"Episodes per approach: {args.episodes}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {args.output}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Train DQN
    set_seed(args.seed)
    dqn_metrics = train_approach(
        'dqn', args.episodes, track_seed=args.seed,
        render=args.render, log_freq=args.log_freq
    )
    
    # Train JEPA (same seed for fair comparison)
    set_seed(args.seed)
    jepa_metrics = train_approach(
        'jepa', args.episodes, track_seed=args.seed,
        render=args.render, log_freq=args.log_freq
    )
    
    # Save results
    summary = save_results(dqn_metrics, jepa_metrics, args.output)
    
    # Print comparison
    print_comparison(summary['dqn'], summary['jepa'])
    
    # Generate plots
    if args.plot and not args.no_plot:
        generate_plots(dqn_metrics, jepa_metrics, args.output)


if __name__ == "__main__":
    main()
