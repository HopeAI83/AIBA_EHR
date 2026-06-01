# AIBA: AI-Accelerated Blockchain Architecture for EHR Exchange

AIBA is a Python-based research prototype for secure and efficient Electronic Healthcare Record (EHR) access-event exchange using three integrated components:

1. **TCN+AM anomaly detection** for identifying abnormal EHR access behaviour from synthetic audit-event sequences.
2. **ChainSimEnv blockchain simulator** for evaluating throughput, latency, consensus behaviour, mempool pressure, and orphan-block rate under simulated workloads.
3. **PPO-based adaptive controller** for adjusting blockchain operating parameters in response to observed system conditions.

The repository is kept as a single-folder package. All code, data, result tables, and figures are stored directly in the repository root for simple upload and inspection.

## Core components

| Component | File(s) |
|---|---|
| ChainSimEnv blockchain simulator | `chainsim_env.py` |
| PPO adaptive controller | `ppo_agent.py` |
| TCN+AM anomaly detection model | `tcn_attention_model.py` |
| Synthetic EHR access-event dataset | `synthetic_ehr_access_events_120000.csv` |
| Result reproduction script | `reproduce_all_results.py` |
| Repository verification script | `verify_repository_readiness.py` |

## Dataset

The repository includes a deterministic synthetic EHR access-event dataset:

- File: `synthetic_ehr_access_events_120000.csv`
- Rows: 120,000
- Label column: `is_anomaly`
- Normal events: 60,000
- Anomalous events: 60,000
- Feature description: `data_dictionary.csv`
- Cross-validation assignment file: `cv_fold_assignments.csv`
- Checksum manifest: `SHA256SUMS.txt`

No real patient data, protected health information, identifiable health information, or human-subject data are included.

## Main code files

- `config.py` — global seed, TCN, PPO, reward, and simulation configuration.
- `data_generator.py` — deterministic synthetic data generator.
- `preprocessing.py` — preprocessing utilities for categorical and numeric features.
- `tcn_attention_model.py` — Temporal Convolutional Network with attention module.
- `chainsim_env.py` — blockchain simulation environment.
- `ppo_agent.py` — PPO controller configuration and wrapper.
- `evaluation.py` — classification metrics and McNemar-test utilities.
- `reproduce_all_results.py` — regenerates principal result tables and figures.
- `verify_repository_readiness.py` — checks required files and dataset integrity.

## Result tables

The repository includes CSV summaries for:

- AI/anomaly-detection performance
- 5-fold cross-validation metrics
- Confusion matrices
- Blockchain throughput and latency
- Ablation analysis
- Runtime/complexity comparison
- PPO hyperparameters
- TCN+AM hyperparameters
- Reward weights
- Attack-resilience analysis
- PBFT/PoS comparison

## Figures

The repository includes PNG figures for:

- Confusion matrices
- ROC curves
- F1-score comparison
- Blockchain throughput
- Blockchain p99 latency
- Ablation performance
- PPO training convergence
- Dataset role distribution
- Dataset anomaly distribution by hour

## Installation

```bash
pip install -r requirements.txt
```

## Verification

Run:

```bash
python verify_repository_readiness.py
```

Expected output:

```text
Repository readiness check PASSED
Dataset rows: 120000
```

## Reproducing selected outputs

Run:

```bash
python reproduce_all_results.py
```

This regenerates the principal result tables and figures in the same folder.

## License

This project is released under the MIT License. The included synthetic dataset is provided for research and reproducibility use only.
