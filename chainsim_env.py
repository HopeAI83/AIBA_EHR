
"""ChainSimEnv: lightweight blockchain simulation environment.

The environment follows the Gymnasium interface when Gymnasium is installed.
It models Poisson transaction arrivals, mempool pressure, validation latency,
orphan block probability, and PPO-controlled block size/difficulty actions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from config import GLOBAL_SEED, REWARD_WEIGHTS, LATENCY_SLO_TARGET_MS

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # lightweight fallback for environments without Gymnasium
    gym = object
    spaces = None

@dataclass
class SimChainParams:
    max_mempool: int = 30_000
    base_arrival_rate: float = 55.0
    min_block_size: int = 200
    max_block_size: int = 1200
    min_difficulty: float = 0.5
    max_difficulty: float = 2.0
    episode_length: int = 3600
    seed: int = GLOBAL_SEED


def compute_reward(tps: float, latency_ms: float, mempool_fraction: float, orphan_rate: float,
                   difficulty_multiplier: float, high_risk_fraction: float = 0.0,
                   weights: dict = REWARD_WEIGHTS, lat_target: float = LATENCY_SLO_TARGET_MS) -> float:
    return (
        weights['w_tps'] * (tps / 1000.0)
        - weights['w_latency'] * max(0.0, latency_ms - lat_target) / 1000.0
        - weights['w_mempool'] * mempool_fraction
        - weights['w_orphan'] * orphan_rate
        - weights['w_difficulty'] * abs(difficulty_multiplier - 1.0)
        - weights.get('w_risk', 0.0) * high_risk_fraction
    )

class ChainSimEnv(gym.Env if hasattr(gym, 'Env') else object):
    metadata = {'render_modes': []}
    def __init__(self, params: SimChainParams | None = None):
        self.params = params or SimChainParams()
        self.rng = np.random.default_rng(self.params.seed)
        if spaces is not None:
            self.observation_space = spaces.Box(low=np.array([0,0,0,0], dtype=np.float32), high=np.array([1,2,2,2.5], dtype=np.float32))
            self.action_space = spaces.Box(low=np.array([self.params.min_block_size,self.params.min_difficulty], dtype=np.float32),
                                           high=np.array([self.params.max_block_size,self.params.max_difficulty], dtype=np.float32))
        self.reset(seed=self.params.seed)
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t = 0; self.mempool = 0.0; self.latency_ewma = 25.0; self.tps_ewma = 50.0; self.orphans = 0
        return self._state(), {}
    def _state(self):
        mem = np.clip(self.mempool / self.params.max_mempool, 0, 1)
        lat = np.clip(self.latency_ewma / 60.0, 0, 2)
        tps = np.clip(self.tps_ewma / 250.0, 0, 2)
        cong = np.clip(0.8 + 0.6*np.sin(2*np.pi*self.t/900) + 0.6*mem, 0, 2.5)
        return np.array([mem, lat, tps, cong], dtype=np.float32)
    def step(self, action):
        block_size = float(np.clip(action[0], self.params.min_block_size, self.params.max_block_size))
        difficulty = float(np.clip(action[1], self.params.min_difficulty, self.params.max_difficulty))
        congestion = self._state()[3]
        arrivals = self.rng.poisson(self.params.base_arrival_rate * congestion)
        self.mempool = min(self.params.max_mempool, self.mempool + arrivals)
        capacity = min(self.mempool, block_size / max(difficulty, 1e-6))
        # latency rises with congestion and difficulty; validated capacity reduces queue
        latency_s = 8.0 + 0.035*block_size + 4.0*difficulty + 20.0*(self.mempool/self.params.max_mempool) + self.rng.normal(0, 2.0)
        latency_s = float(np.clip(latency_s, 0.1, 1500.0))
        orphan_p = float(np.clip(0.001 + 0.0006*latency_s + 0.01*(difficulty>1.6), 0, 0.25))
        orphan = self.rng.random() < orphan_p
        if not orphan:
            self.mempool = max(0.0, self.mempool - capacity)
        else:
            self.orphans += 1
        tps = capacity if not orphan else 0.0
        self.latency_ewma = 0.9*self.latency_ewma + 0.1*latency_s
        self.tps_ewma = 0.9*self.tps_ewma + 0.1*tps
        high_risk_fraction = float(np.clip(0.05 + 0.25*(congestion > 1.2) + self.rng.normal(0,0.03), 0, 1))
        reward = compute_reward(tps=tps, latency_ms=latency_s*1000, mempool_fraction=self.mempool/self.params.max_mempool,
                                orphan_rate=orphan_p, difficulty_multiplier=difficulty, high_risk_fraction=high_risk_fraction)
        self.t += 1
        terminated = self.t >= self.params.episode_length
        info = {'tps': tps, 'latency_s': latency_s, 'orphan_rate': orphan_p, 'block_size': block_size,
                'difficulty': difficulty, 'mempool': self.mempool, 'high_risk_fraction': high_risk_fraction}
        return self._state(), float(reward), terminated, False, info
