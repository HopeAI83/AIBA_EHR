
"""Central configuration for the AIBA-EHR reproducibility release."""
from __future__ import annotations

GLOBAL_SEED = 42
DATASET_ROWS = 120_000
SEQUENCE_LENGTH = 20
TCN_DILATION_RATES = [1, 2, 4, 8]
TCN_KERNEL_SIZE = 3
TCN_FILTERS = 64
TCN_ATTENTION_HEADS = 4
TCN_ATTENTION_DIM = 64
TCN_DROPOUT = 0.2
TCN_RECEPTIVE_FIELD = 1 + (TCN_KERNEL_SIZE - 1) * sum(TCN_DILATION_RATES)
assert TCN_RECEPTIVE_FIELD == 31
assert TCN_RECEPTIVE_FIELD >= SEQUENCE_LENGTH

PPO_CONFIG = {
    "algorithm": "PPO",
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.995,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "total_timesteps": 100_000,
    "policy": "MlpPolicy",
    "seed": GLOBAL_SEED,
}

REWARD_WEIGHTS = {
    "w_tps": 0.40,
    "w_latency": 0.30,
    "w_mempool": 0.10,
    "w_orphan": 0.15,
    "w_difficulty": 0.05,
    "w_risk": 0.10,
}
LATENCY_SLO_TARGET_MS = 30.0
RISK_THRESHOLD = 0.70
