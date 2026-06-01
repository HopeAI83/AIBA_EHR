
"""Preprocessing and sequence construction for EHR access events."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'user_role','department','resource_type','access_method','geo_region','device_type','auth_level','access_reason',
    'hour_of_day','day_of_week','request_rate','failed_attempts','ip_signature_hash','response_time_ms','data_volume_kb'
]
CATEGORICAL_COLUMNS = ['user_role','department','resource_type','access_method','geo_region','device_type','auth_level','access_reason']
NUMERIC_COLUMNS = ['hour_of_day','day_of_week','request_rate','failed_attempts','ip_signature_hash','response_time_ms','data_volume_kb']


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    x = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        x[col] = x[col].astype('category').cat.codes.astype(float)
    x[NUMERIC_COLUMNS] = x[NUMERIC_COLUMNS].astype(float)
    scaler = StandardScaler()
    x.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(x[FEATURE_COLUMNS])
    return x, scaler


def build_sliding_windows(df: pd.DataFrame, sequence_length: int = 20, step: int = 1):
    x_df, scaler = encode_features(df)
    y = df['is_anomaly'].to_numpy(dtype=int)
    values = x_df.to_numpy(dtype=np.float32)
    xs, ys = [], []
    for start in range(0, len(df) - sequence_length + 1, step):
        end = start + sequence_length
        xs.append(values[start:end])
        ys.append(y[end-1])
    return np.stack(xs), np.asarray(ys, dtype=np.int64), scaler
