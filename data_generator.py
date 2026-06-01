
"""Synthetic EHR access-event data generator.

This script generates the public 120,000-row synthetic EHR access-event dataset
used in the reproducibility package. It does not use any real patient data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import GLOBAL_SEED, DATASET_ROWS


def generate_synthetic_ehr_access_events(n: int = DATASET_ROWS, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = np.datetime64('2026-01-01T00:00:00')
    base_minutes = np.arange(n)
    jitter = rng.integers(0, 60, size=n)
    timestamps = start + base_minutes.astype('timedelta64[m]') + jitter.astype('timedelta64[s]')
    hours = pd.to_datetime(timestamps.astype(str)).hour.values
    day_of_week = pd.to_datetime(timestamps.astype(str)).dayofweek.values

    roles = np.array(['Physician','Nurse','Admin','Radiology','LabTech','Charge'])
    user_role = rng.choice(roles, size=n, p=np.array([0.28,0.26,0.09,0.12,0.15,0.10]))
    department = rng.choice(np.array(['Emergency','Cardiology','Radiology','Laboratory','Oncology','Pediatrics','ICU','General Medicine']), size=n,
                            p=np.array([0.15,0.13,0.12,0.15,0.10,0.10,0.10,0.15]))
    resource_type = rng.choice(np.array(['MedicationRecord','LabResult','ImagingReport','ClinicalNote','DischargeSummary','BillingRecord','AllergyRecord','VitalSigns']), size=n,
                               p=np.array([0.16,0.18,0.12,0.18,0.10,0.08,0.08,0.10]))
    access_method = rng.choice(np.array(['web_portal','mobile_app','api_gateway','ehr_terminal','batch_export']), size=n,
                               p=np.array([0.34,0.18,0.21,0.24,0.03]))
    geo_region = rng.choice(np.array(['hospital_lan','city_clinic','regional_center','remote_vpn','foreign_ip']), size=n,
                            p=np.array([0.52,0.20,0.14,0.11,0.03]))
    device_type = rng.choice(np.array(['workstation','tablet','mobile','diagnostic_terminal','unknown_device']), size=n,
                             p=np.array([0.49,0.13,0.15,0.18,0.05]))
    auth_level = rng.choice(np.array(['password','mfa','sso','break_glass']), size=n,
                            p=np.array([0.38,0.34,0.24,0.04]))
    access_reason = rng.choice(np.array(['treatment','diagnosis','lab_review','emergency','administration','research_audit','billing']), size=n,
                               p=np.array([0.34,0.18,0.16,0.08,0.10,0.06,0.08]))

    working_hours = ((hours >= 8) & (hours <= 18)).astype(float)
    night_hours = ((hours < 6) | (hours > 22)).astype(float)
    base_rate = 2.0 + 2.8*working_hours + 0.4*np.sin(2*np.pi*(hours-8)/24)
    request_rate = rng.gamma(shape=2.2, scale=base_rate/2.2, size=n)
    failed_attempts = rng.poisson(lam=0.12 + 0.35*night_hours + 0.18*(auth_level=='password'), size=n)
    ip_signature_hash = rng.integers(10_000, 999_999, size=n)
    response_time_ms = np.clip(rng.normal(loc=180 + 60*(access_method=='api_gateway') + 90*(geo_region=='remote_vpn') + 210*(geo_region=='foreign_ip'), scale=45, size=n), 20, None)
    data_volume_kb = np.clip(rng.lognormal(mean=6.0 + 0.8*(resource_type=='ImagingReport') + 0.5*(access_method=='batch_export'), sigma=0.55, size=n), 10, None)
    risk = (0.85*night_hours + 0.55*(geo_region=='foreign_ip') + 0.40*(geo_region=='remote_vpn') + 0.50*(device_type=='unknown_device') +
            0.55*(auth_level=='break_glass') + 0.65*(access_method=='batch_export') + 0.35*(access_reason=='research_audit') +
            0.50*(failed_attempts >= 2) + 0.30*(request_rate > np.quantile(request_rate, 0.85)) +
            0.25*(data_volume_kb > np.quantile(data_volume_kb, 0.90)) + rng.normal(0, 0.45, size=n))
    idx_sorted = np.argsort(risk)[::-1]
    is_anomaly = np.zeros(n, dtype=int)
    is_anomaly[idx_sorted[:n//2]] = 1
    response_time_ms = np.round(response_time_ms + is_anomaly*rng.normal(55, 20, size=n), 2)
    data_volume_kb = np.round(data_volume_kb * (1 + is_anomaly*rng.uniform(0.05, 0.35, size=n)), 2)
    request_rate = np.round(request_rate + is_anomaly*rng.uniform(0.2, 1.2, size=n), 3)
    return pd.DataFrame({
        'event_id': np.arange(1, n+1),
        'event_timestamp': pd.to_datetime(timestamps.astype(str)).astype(str),
        'user_role': user_role,
        'department': department,
        'resource_type': resource_type,
        'access_method': access_method,
        'geo_region': geo_region,
        'device_type': device_type,
        'auth_level': auth_level,
        'access_reason': access_reason,
        'hour_of_day': hours,
        'day_of_week': day_of_week,
        'request_rate': request_rate,
        'failed_attempts': failed_attempts,
        'ip_signature_hash': ip_signature_hash,
        'response_time_ms': response_time_ms,
        'data_volume_kb': data_volume_kb,
        'is_anomaly': is_anomaly,
    })


if __name__ == '__main__':
    df = generate_synthetic_ehr_access_events()
    df.to_csv('synthetic_ehr_access_events_120000.csv', index=False)
    print(df.shape)
