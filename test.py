#!/usr/bin/env python3
"""
Test and evaluation script for the RL racing agent.

This script provides various test modes:
- Test a trained agent
- Compare random vs trained agent
- Evaluate on multiple tracks
"""

import argparse
import os
import random
import numpy as np
import json
import csv
from datetime import datetime
import torch

from game.track import Track
from rl.agent import DQNAgent, RandomAgent
from rl.environment import RacingEnv, MultiTrackEnv
from jepa.agent import JEPAAgent
from evolution.agent import EvolutionAgent
from utils.config import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test RL agent for racing game')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the test')
    parser.add_argument('--track', type=int, default=None,
                        help='Specific track index to test on')
    parser.add_argument('--compare', action='store_true',
                        help='Compare random vs trained agent')
    parser.add_argument('--multi-track', action='store_true',
                        help='Test on multiple tracks')
    parser.add_argument('--num-tracks', type=int, default=5,
                        help='Number of tracks for multi-track testing')
    parser.add_argument('--output', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--record', action='store_true',
                        help='Record test results to file')
    parser.add_argument('--render-every-n', type=int, default=1,
                        help='Render every Nth step when rendering')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_SEED,
                        help='Random seed for reproducible runs')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic torch algorithms where available')
    parser.add_argument('--approach', type=str, default=config.DEFAULT_APPROACH,
                        choices=['dqn', 'jepa', 'evo'],
                        help='Learning approach: dqn, jepa, or evo')
    
    return parser.parse_args()


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy, and torch random sources."""
    # Evaluation also needs fixed RNGs so score changes reflect model changes, not new tracks/noise.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TestResult:
    """Container for test results."""
    
    def __init__(self):
        self.episode_rewards = []
        self.lap_times = []
        self.completed_laps = 0
        self.total_episodes = 0
        self.collision_count = 0
        self.avg_progress = 0
        self.total_steps = 0
        self.records = []  # list of dicts with per-episode data
    
    def add_episode(self, reward, lap_time, completed_lap, collision_count, progress, steps):
        """Add results from one episode."""
        self.episode_rewards.append(reward)
        self.lap_times.append(lap_time)
        if completed_lap:
            self.completed_laps += 1
        self.collision_count += collision_count
        self.avg_progress += progress
        self.total_episodes += 1
        self.total_steps += steps
        
        # Store per-episode record
        self.records.append({
            'reward': reward,
            'lap_time': lap_time,
            'completed_lap': completed_lap,
            'collision_count': collision_count,
            'progress': progress,
            'steps': steps,
        })
    
    def get_summary(self):
        """Get summary statistics."""
        # Reward is useful, but lap completion and collision rate often tell policy quality more directly.
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        std_reward = np.std(self.episode_rewards) if self.episode_rewards else 0
        avg_lap_time = np.mean(self.lap_times) if self.lap_times else 0
        min_lap_time = np.min(self.lap_times) if self.lap_times else 0
        max_lap_time = np.max(self.lap_times) if self.lap_times else 0
        lap_completion_rate = (self.completed_laps / self.total_episodes * 100) if self.total_episodes > 0 else 0
        avg_progress = (self.avg_progress / self.total_episodes * 100) if self.total_episodes > 0 else 0
        avg_steps = self.total_steps / self.total_episodes if self.total_episodes > 0 else 0
        
        return {
            'total_episodes': self.total_episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_lap_time': avg_lap_time,
            'min_lap_time': min_lap_time,
            'max_lap_time': max_lap_time,
            'lap_completion_rate': lap_completion_rate,
            'avg_progress': avg_progress,
            'avg_steps_per_episode': avg_steps,
            'collision_rate': self.collision_count / self.total_episodes if self.total_episodes > 0 else 0
        }
    
    def print_summary(self, name="Agent"):
        """Print summary of results."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print(f"{name} Test Results")
        print("=" * 60)
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Average reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Lap completion rate: {summary['lap_completion_rate']:.1f}%")
        print(f"Average lap time: {summary['avg_lap_time']:.2f}s")
        print(f"Best lap time: {summary['min_lap_time']:.2f}s")
        print(f"Worst lap time: {summary['max_lap_time']:.2f}s")
        print(f"Average progress: {summary['avg_progress']:.1f}%")
        print(f"Average steps per episode: {summary['avg_steps_per_episode']:.0f}")
        print(f"Collision rate: {summary['collision_rate']:.1f}%")
        print("=" * 60)


def test_agent(agent, env, num_episodes, render=False, name="Agent"):
    """
    Test an agent on the environment.
    
    Args:
        agent: DQNAgent or RandomAgent - agent to test
        env: RacingEnv or MultiTrackEnv - environment to test on
        num_episodes: int - number of episodes to test
        render: bool - whether to render
        name: str - name of the agent for display
    
    Returns:
        TestResult: test results
    """
    result = TestResult()
    
    print(f"\nTesting {name} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        lap_time = 0
        lap_completed = False
        collision_count = 0
        total_progress = 0
        
        done = False
        steps = 0
        
        while not done:
            # Testing disables exploration so results show learned policy, not random epsilon actions.
            action = agent.select_action(state, explore=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Track progress
            total_progress += info.get('progress', 0)
            
            # Check for collisions
            if info.get('collision', False):
                collision_count += 1
            
            # Check for lap completion
            if info.get('lap_completed', False):
                lap_time = info.get('lap_time', 0)
                lap_completed = True
            
            state = next_state
            
            # Render
            if render:
                env.render()
        
        # Calculate average progress
        avg_progress = total_progress / max(1, steps)
        
        # Add to results
        result.add_episode(
            reward=episode_reward,
            lap_time=lap_time if lap_completed else 0,
            completed_lap=lap_completed,
            collision_count=collision_count,
            progress=avg_progress,
            steps=steps
        )
        
        # Print episode results
        lap_str = f"{lap_time:.2f}s" if lap_completed else "N/A"
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Lap={lap_str}, Progress={avg_progress*100:.1f}%",
              f"Collisions={collision_count}")
    
    return result


def compare_agents(args):
    """
    Compare random agent vs trained agent.
    
    Args:
        args: command line arguments
    """
    print("\n" + "=" * 60)
    print("Comparing Random vs Trained Agent")
    print("=" * 60)
    
    # Create environment
    track = Track(seed=args.seed) if args.seed is not None else None
    env = RacingEnv(track=track, render=args.render, render_every_n=args.render_every_n)
    
    # Random agent is baseline: if trained policy cannot beat this, learning pipeline needs work.
    random_agent = RandomAgent(config.ACTION_DIM)
    random_result = test_agent(
        random_agent, env, args.episodes,
        render=args.render, name="Random Agent"
    )
    
    # Same environment and metrics keep comparison fair across baseline and learned policy.
    trained_result = None
    if args.model and os.path.exists(args.model):
        if args.approach == 'jepa':
            trained_agent = JEPAAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
            )
        elif args.approach == 'evo':
            trained_agent = EvolutionAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
            )
        else:
            trained_agent = DQNAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
                epsilon=0.0  # No exploration
            )
        trained_agent.load(args.model)
        trained_result = test_agent(
            trained_agent, env, args.episodes,
            render=args.render, name="Trained Agent"
        )
        
        # Compare completion, time, and reward together; one metric alone can hide brittle behavior.
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        
        random_summary = random_result.get_summary()
        trained_summary = trained_result.get_summary()
        
        print(f"Lap completion rate:")
        print(f"  Random: {random_summary['lap_completion_rate']:.1f}%")
        print(f"  Trained: {trained_summary['lap_completion_rate']:.1f}%")
        print(f"  Improvement: {trained_summary['lap_completion_rate'] - random_summary['lap_completion_rate']:+.1f}%")
        
        print(f"\nAverage lap time:")
        print(f"  Random: {random_summary['avg_lap_time']:.2f}s")
        print(f"  Trained: {trained_summary['avg_lap_time']:.2f}s")
        if random_summary['avg_lap_time'] > 0:
            improvement = (random_summary['avg_lap_time'] - trained_summary['avg_lap_time']) / random_summary['avg_lap_time'] * 100
            print(f"  Improvement: {improvement:+.1f}%")
        
        print(f"\nAverage reward:")
        print(f"  Random: {random_summary['avg_reward']:.2f}")
        print(f"  Trained: {trained_summary['avg_reward']:.2f}")
        print(f"  Improvement: {trained_summary['avg_reward'] - random_summary['avg_reward']:+.2f}")
    else:
        print("No model specified for trained agent.")
    
    env.close()
    
    # Save results if requested
    if args.record:
        results = (random_result, trained_result) if trained_result is not None else (random_result,)
        _save_results(args, *results)


def test_multi_track(args):
    """
    Test agent on multiple tracks.
    
    Args:
        args: command line arguments
    """
    print(f"\nTesting on {args.num_tracks} tracks...")
    
    # Multi-track eval checks generalization instead of memorization on one lucky procedural track.
    env = MultiTrackEnv(
        num_tracks=args.num_tracks,
        render=args.render,
        eval_mode=True,
        render_every_n=args.render_every_n,
        seed=args.seed,
    )
    
    # Create and load agent
    if args.model and os.path.exists(args.model):
        if args.approach == 'jepa':
            agent = JEPAAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
            )
        elif args.approach == 'evo':
            agent = EvolutionAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
            )
        else:
            agent = DQNAgent(
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
                epsilon=0.0
            )
        agent.load(args.model)
    else:
        print("No model specified. Using random agent.")
        agent = RandomAgent(config.ACTION_DIM)
    
    # More tracks widen distribution of states, making summary metrics harder to game.
    result = test_agent(agent, env, args.episodes * args.num_tracks,
                       render=args.render, name="Multi-Track Test")
    
    result.print_summary("Multi-Track Test")
    
    env.close()
    
    # Save results if requested
    if args.record:
        _save_results(args, result)


def _save_results(args, *results):
    """
    Save test results to file.
    
    Args:
        args: command line arguments
        results: TestResult objects to save
    """
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    csv_path = os.path.join(args.output, f'test_results_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Agent', 'Episode', 'Reward', 'Lap Time', 'Completed', 'Progress', 'Collisions'])
        
        for i, result in enumerate(results):
            name = f"Agent_{i}" if len(results) > 1 else "Agent"
            for j, record in enumerate(result.records):
                writer.writerow([
                    name, 
                    j + 1, 
                    record['reward'],
                    record['lap_time'],
                    record['completed_lap'],
                    record['progress'],
                    record['collision_count']
                ])
    
    # Save summary
    json_path = os.path.join(args.output, f'test_summary_{timestamp}.json')
    summary = {}
    for i, result in enumerate(results):
        name = f"Agent_{i}" if len(results) > 1 else "Agent"
        summary[name] = result.get_summary()
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


def main():
    """Main entry point."""
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed, deterministic=args.deterministic)
    
    if args.compare:
        compare_agents(args)
    elif args.multi_track:
        test_multi_track(args)
    else:
        # Simple test
        track = Track(seed=args.seed) if args.seed is not None else None
        env = RacingEnv(track=track, render=args.render, render_every_n=args.render_every_n)
        
        if args.model and os.path.exists(args.model):
            if args.approach == 'jepa':
                agent = JEPAAgent(
                    state_dim=config.STATE_DIM,
                    action_dim=config.ACTION_DIM,
                )
            elif args.approach == 'evo':
                agent = EvolutionAgent(
                    state_dim=config.STATE_DIM,
                    action_dim=config.ACTION_DIM,
                )
            else:
                agent = DQNAgent(
                    state_dim=config.STATE_DIM,
                    action_dim=config.ACTION_DIM,
                    epsilon=0.0
                )
            agent.load(args.model)
            name = f"Trained Agent ({args.approach.upper()})"
        else:
            agent = RandomAgent(config.ACTION_DIM)
            name = "Random Agent"
        
        result = test_agent(agent, env, args.episodes,
                           render=args.render, name=name)
        result.print_summary(name)
        
        env.close()
        
        if args.record:
            _save_results(args, result)


if __name__ == "__main__":
    main()
