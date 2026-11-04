"""
Feature engineering module.
Transforms patient state into the feature vector required by the classifier.
"""

import numpy as np
from src.aki.state import Patient
from src.aki.config import FEATURE_ORDER

SECONDS_PER_DAY = 86400.0
MIN_TIME_DELTA = 0.0001     # Avoid div by zero


def compute_features(patient: Patient) -> np.ndarray:
    """
    Generates a 1x10 feature vector for inference.

    Features:
        - age,
        - is_female
        - baseline (min)
        - current
        - peak
        - volatility (std dev)
        - count
        - velocity (rate of change per day)
        - ratio (current / baseline)
        - abs_increase (current - baseline)
    """
    if not patient.history:
        return np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)

    # (1) data extraction
    all_values = [r.value for r in patient.history]
    current_record = patient.history[-1]
    current_value = current_record.value
    current_date = current_record.date

    # (2) basic stats
    baseline = min(all_values)
    peak = max(all_values)
    count = len(all_values)

    # (3) volatility (std dev)
    # Use ddof = 1 (sample std dev) to match training (pandas default)
    if count > 1:
        volatility = np.std(all_values, ddof=1)
    else:
        volatility = 0.0

    # (4) velocity (rate of change per day)
    velocity = 0.0
    if count > 1:
        prev_record = patient.history[-2]
        delta_days = (current_date - prev_record.date).total_seconds() / SECONDS_PER_DAY
        if delta_days > MIN_TIME_DELTA:
            value_delta = current_value - prev_record.value
            velocity = value_delta / delta_days

    # (5) derived features
    ratio = (current_value / baseline) if baseline > 0 else 0.
    abs_increase = current_value - baseline

    # (6) the feature vector must match the training order
    data = dict(
        age=patient.age,
        is_female=patient.is_female,
        baseline=baseline,
        current=current_value,
        peak=peak,
        volatility=volatility,
        count=count,
        velocity=velocity,
        ratio=ratio,
        abs_increase=abs_increase
    )
    vector = [data[feature_name] for feature_name in FEATURE_ORDER]
    return np.array([vector], dtype=np.float32)