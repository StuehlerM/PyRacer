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
import glob
import os
import random
import time
import json
import csv
from datetime import datetime
import numpy as np
import torch

from game.track import Track
from game.environment import RacingEnv, MultiTrackEnv
from rl.agent import DQNAgent, RandomAgent
from jepa.agent import JEPAAgent
from evolution.agent import EvolutionAgent
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
    parser.add_argument('--load-best', action='store_true',
                        help='Load latest best model for the selected approach from save-dir')
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
                        choices=['dqn', 'jepa', 'evo'],
                        help='Learning approach: dqn (reward-based RL), jepa (self-supervised world model), or evo (neuroevolution)')
    
    # --- Evolution (neuroevolution) options ---
    evo = parser.add_argument_group('evolution (--approach evo)')
    evo.add_argument('--generations', type=int, default=config.EVO_GENERATIONS,
                     help='Number of generations to evolve (evolution only)')
    evo.add_argument('--pop-size', type=int, default=config.EVO_POP_SIZE,
                     help='Population size: policies per generation (evolution only)')
    evo.add_argument('--eval-episodes', type=int, default=config.EVO_EVAL_EPISODES,
                     help='Episodes averaged to score one policy (evolution only)')
    evo.add_argument('--evo-hidden-dim', type=int, default=config.EVO_HIDDEN_DIM,
                     help='Hidden width of each evolved policy network (evolution only)')
    evo.add_argument('--mutation-std', type=float, default=config.EVO_MUTATION_STD,
                     help='Initial Gaussian mutation std (evolution only)')
    evo.add_argument('--elite-frac', type=float, default=config.EVO_ELITE_FRAC,
                     help='Fraction of top policies carried over unchanged (evolution only)')
    evo.add_argument('--tournament-size', type=int, default=config.EVO_TOURNAMENT_SIZE,
                     help='Candidates per selection tournament (evolution only)')
    evo.add_argument('--crossover-rate', type=float, default=config.EVO_CROSSOVER_RATE,
                     help='Probability of crossover vs cloning a parent (evolution only)')
    
    args = parser.parse_args()
    if args.load and args.load_best:
        parser.error('--load and --load-best cannot be used together')
    return args


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


def _best_model_timestamp(model_path):
    """Extract timestamp portion from a best_model_<timestamp>.pth path."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    prefix = 'best_model_'
    if not stem.startswith(prefix):
        return ''
    return stem[len(prefix):]


def _run_config_path_for_model(model_path, log_dir):
    """Return matching config_<timestamp>.json path for a best model."""
    timestamp = _best_model_timestamp(model_path)
    if not timestamp:
        return None
    return os.path.join(log_dir, f'config_{timestamp}.json')


def _model_approach_from_run_config(model_path, log_dir):
    """Read approach metadata saved beside the run logs."""
    config_path = _run_config_path_for_model(model_path, log_dir)
    if config_path is None or not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        run_config = json.load(f)
    args = run_config.get('args', {})
    return args.get('approach', config.DEFAULT_APPROACH)


def find_latest_best_model(save_dir, log_dir, approach):
    """Find newest best model whose saved run config matches approach."""
    pattern = os.path.join(save_dir, 'best_model_*.pth')
    candidates = sorted(
        glob.glob(pattern),
        key=lambda path: (_best_model_timestamp(path), os.path.getmtime(path)),
        reverse=True,
    )

    for model_path in candidates:
        if _model_approach_from_run_config(model_path, log_dir) == approach:
            return model_path
    return None


def load_model_for_run(agent, args):
    """Load explicit model path or latest best model requested by CLI args."""
    if args.load_best:
        load_path = find_latest_best_model(args.save_dir, args.log_dir, args.approach)
        if load_path is None:
            print(f"No previous best {args.approach.upper()} model found in {args.save_dir}; starting fresh.")
            return None
        print(f"Loading latest best {args.approach.upper()} model from {load_path}...")
    elif args.load:
        if not os.path.exists(args.load):
            print(f"Model path not found: {args.load}. Starting fresh.")
            return None
        load_path = args.load
        print(f"Loading model from {load_path}...")
    else:
        return None

    agent.load(load_path)
    print("Model loaded!")
    return load_path


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
                'best_lap_time', 'epsilon', 'loss', 'steps', 'memory_size',
                'max_progress', 'collision', 'off_track', 'stalled'
            ])
    
    def log_episode(self, episode, total_reward, avg_reward, lap_time, 
                    best_lap_time, epsilon, loss, steps, memory_size,
                    max_progress=0.0, collision=False, off_track=False,
                    stalled=False):
        """Log an episode to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, avg_reward, lap_time,
                best_lap_time, epsilon, loss, steps, memory_size,
                max_progress, collision, off_track, stalled
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
                      best_lap_time, epsilon, loss, steps, max_progress=0.0):
        """Print training summary for an episode."""
        print(f"\rEpisode {episode:5d} | "
              f"Reward: {total_reward:8.2f} | "
              f"Avg: {avg_reward:8.2f} | "
              f"MaxProg: {max_progress * 100:5.1f}% | "
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
    # Neuroevolution has a fundamentally different (generational) training loop,
    # so it gets its own routine rather than the per-step interaction loop below.
    if args.approach == 'evo':
        return train_evolution(args)
    
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
    
    load_model_for_run(agent, args)
    
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
            episode_max_progress = 0.0
            episode_collision = False
            episode_off_track = False
            episode_stalled = False
            
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
                episode_max_progress = max(
                    episode_max_progress,
                    info.get('forward_progress', info.get('progress', 0.0)),
                )
                episode_collision = episode_collision or info.get('collision', False)
                episode_off_track = episode_off_track or info.get('off_track', False)
                episode_stalled = episode_stalled or info.get('stalled', False)
                
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
                memory_size=len(agent.memory),
                max_progress=episode_max_progress,
                collision=episode_collision,
                off_track=episode_off_track,
                stalled=episode_stalled
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
                    steps=agent_stats['env_steps'],
                    max_progress=episode_max_progress
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


def _evaluate_genome(env, agent, genome, episodes, render=False):
    """
    Score one genome (policy) by letting it drive for a few episodes.

    Fitness is the mean episode return — the same "drive far and fast" objective
    DQN optimizes, but here it is just a black-box score handed to the genetic
    algorithm. No gradients flow; the policy weights came straight from the genome.

    Args:
        env: Racing environment.
        agent: EvolutionAgent (its policy is set to this genome).
        genome: Flat weight vector to evaluate.
        episodes: Episodes to average over.
        render: Whether to render during evaluation.

    Returns:
        dict with mean fitness, lap stats and max progress.
    """
    agent.set_active_genome(genome)

    returns = []
    laps_completed = 0
    best_lap_time = float('inf')
    max_progress = 0.0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            agent.on_env_step()

            max_progress = max(max_progress, info.get('progress', 0))
            if info.get('lap_completed', False):
                laps_completed += 1
                lap_time = info.get('lap_time', 0)
                if 0 < lap_time < best_lap_time:
                    best_lap_time = lap_time

            if render:
                env.render()
        returns.append(episode_reward)

    return {
        'fitness': float(np.mean(returns)) if returns else 0.0,
        'laps_completed': laps_completed,
        'best_lap_time': best_lap_time,
        'max_progress': max_progress,
    }


def train_evolution(args):
    """
    Train a policy with neuroevolution (a genetic algorithm).

    Unlike DQN/JEPA, there is no per-step gradient update. Each generation we
    score every policy in the population by driving, then breed the fittest into
    the next generation (elitism + tournament selection + crossover + mutation).

    Args:
        args: argparse.Namespace - command line arguments
    """
    print("Starting training...")
    print(f"Approach: EVO (neuroevolution / genetic algorithm)")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Episodes per policy: {args.eval_episodes}")
    print(f"Render: {args.render}")
    print(f"Multi-track: {args.multi_track}")
    print()

    logger = TrainingLogger(args.log_dir, args.save_dir)
    logger.save_config(args)

    # Same environment options as the RL path, so comparisons stay fair.
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

    agent = EvolutionAgent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=args.evo_hidden_dim,
        pop_size=args.pop_size,
        elite_frac=args.elite_frac,
        mutation_std=args.mutation_std,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
    )
    print(f"Using neuroevolution (gradient-free, {agent.num_params} weights per policy)")
    print(f"  Selection: tournament (size {args.tournament_size}) + {agent.population.num_elites} elites")
    print(f"  Variation: uniform crossover (p={args.crossover_rate}) + Gaussian mutation (std={args.mutation_std})")
    print()

    # Optionally seed the population's best slot from a saved policy.
    load_model_for_run(agent, args)

    start_time = time.time()
    best_lap_time = float('inf')
    best_fitness_so_far = -float('inf')
    generation = 0

    try:
        for generation in range(1, args.generations + 1):
            genomes = agent.ask()

            fitnesses = []
            gen_laps = 0
            gen_best_lap = float('inf')
            gen_max_progress = 0.0

            # Score every policy in the population.
            for genome in genomes:
                result = _evaluate_genome(
                    env, agent, genome, args.eval_episodes, render=args.render
                )
                fitnesses.append(result['fitness'])
                gen_laps += result['laps_completed']
                gen_best_lap = min(gen_best_lap, result['best_lap_time'])
                gen_max_progress = max(gen_max_progress, result['max_progress'])

            if gen_best_lap < best_lap_time:
                best_lap_time = gen_best_lap

            # Breed the next generation from the fitnesses.
            agent.tell(fitnesses)
            agent.increment_episode()

            gen_max_fitness = float(np.max(fitnesses))
            gen_mean_fitness = float(np.mean(fitnesses))
            stats = agent.get_stats()

            # Save whenever we find a new all-time-best policy.
            if agent.population.best_fitness > best_fitness_so_far:
                best_fitness_so_far = agent.population.best_fitness
                logger.save_model(agent, is_best=True)

            # Reuse the CSV schema: generation plays the role of "episode",
            # max/mean fitness map to total/avg reward, mutation std -> "epsilon".
            logger.log_episode(
                episode=generation,
                total_reward=gen_max_fitness,
                avg_reward=gen_mean_fitness,
                lap_time=gen_best_lap if gen_best_lap != float('inf') else 0,
                best_lap_time=best_lap_time,
                epsilon=stats['mutation_std'],
                loss=0.0,
                steps=stats['env_steps'],
                memory_size=args.pop_size,
            )

            if generation % args.log_freq == 0 or generation == 1:
                lap_str = f"{best_lap_time:.2f}s" if best_lap_time != float('inf') else "N/A"
                print(f"\rGen {generation:5d} | "
                      f"Best fit: {agent.population.best_fitness:8.2f} | "
                      f"Gen max: {gen_max_fitness:8.2f} | "
                      f"Gen avg: {gen_mean_fitness:8.2f} | "
                      f"Laps: {gen_laps:3d} | "
                      f"Progress: {gen_max_progress*100:5.1f}% | "
                      f"Best lap: {lap_str} | "
                      f"MutStd: {stats['mutation_std']:.4f}",
                      end='', flush=True)

            if generation % args.save_freq == 0:
                logger.save_model(agent, is_best=False)

        logger.save_model(agent, is_best=False)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        total_time = time.time() - start_time
        logger.print_final_summary(
            total_episodes=generation,
            total_time=total_time,
            best_lap_time=best_lap_time,
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
    elif args.approach == 'evo':
        agent = EvolutionAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            hidden_dim=args.evo_hidden_dim,
            seed=args.seed,
        )
    else:
        agent = DQNAgent(
            state_dim=config.STATE_DIM,
            action_dim=config.ACTION_DIM,
            lr=args.lr,
            gamma=args.gamma,
            epsilon=0.0,  # No exploration during testing
            epsilon_min=0.0,
            batch_size=args.batch_size
        )
    
    # Load model
    if load_model_for_run(agent, args) is None:
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
