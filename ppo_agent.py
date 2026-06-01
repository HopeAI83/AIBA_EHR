
"""PPO agent training/evaluation utilities for ChainSimEnv."""
from __future__ import annotations
import numpy as np
from chainsim_env import ChainSimEnv
from config import PPO_CONFIG


def heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """Deterministic fallback policy used when Stable-Baselines3 is unavailable."""
    mem, lat, tps, cong = obs
    block_size = 200 + 1000*np.clip(0.45 + 0.35*mem + 0.20*cong/2.5 - 0.10*lat/2, 0, 1)
    difficulty = 1.0 + 0.35*np.clip(cong-1.0, 0, 1.5) - 0.20*np.clip(mem-0.7, 0, 0.3)
    return np.array([block_size, np.clip(difficulty, 0.5, 2.0)], dtype=np.float32)


def train_ppo(total_timesteps: int = PPO_CONFIG['total_timesteps'], seed: int = PPO_CONFIG['seed']):
    """Train PPO using Stable-Baselines3 if installed."""
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError('stable-baselines3 is required for PPO training. Use heuristic_policy for a dependency-light run.') from exc
    env = ChainSimEnv()
    model = PPO('MlpPolicy', env, learning_rate=PPO_CONFIG['learning_rate'], n_steps=PPO_CONFIG['n_steps'],
                batch_size=PPO_CONFIG['batch_size'], n_epochs=PPO_CONFIG['n_epochs'], gamma=PPO_CONFIG['gamma'],
                gae_lambda=PPO_CONFIG['gae_lambda'], clip_range=PPO_CONFIG['clip_range'], ent_coef=PPO_CONFIG['ent_coef'],
                vf_coef=PPO_CONFIG['vf_coef'], seed=seed, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_policy(policy_fn=heuristic_policy, steps: int = 3600) -> dict:
    env = ChainSimEnv(); obs, _ = env.reset()
    metrics=[]
    for _ in range(steps):
        obs, reward, done, _, info = env.step(policy_fn(obs))
        metrics.append(info)
        if done: break
    tps = np.array([m['tps'] for m in metrics], dtype=float)
    lat = np.array([m['latency_s'] for m in metrics], dtype=float)
    return {'average_tps': float(np.mean(tps)), 'peak_tps': float(np.max(tps)), 'latency_mean_s': float(np.mean(lat)),
            'latency_p95_s': float(np.percentile(lat,95)), 'latency_p99_s': float(np.percentile(lat,99))}
