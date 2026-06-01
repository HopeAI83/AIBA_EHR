#!/usr/bin/env python3
"""Minimal repository readiness check for the AIBA-EHR GitHub package."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parent
required = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "synthetic_ehr_access_events_120000.csv",
    "data_dictionary.csv",
    "cv_fold_assignments.csv",
    "SHA256SUMS.txt",
    "chainsim_env.py",
    "ppo_agent.py",
    "tcn_attention_model.py",
    "config.py",
    "ai_module_performance.csv",
    "blockchain_performance_summary.csv",
    "ai_f1_performance_bar.png",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

df = pd.read_csv(ROOT / "synthetic_ehr_access_events_120000.csv")
if len(df) != 120000:
    raise SystemExit(f"Dataset row count mismatch: expected 120000, got {len(df)}")
if "is_anomaly" not in df.columns:
    raise SystemExit("Dataset does not contain is_anomaly label column")
counts = df["is_anomaly"].value_counts().to_dict()
print("Repository readiness check PASSED")
print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")
print(f"Class distribution: {counts}")
