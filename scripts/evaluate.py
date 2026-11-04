"""
Evaluation script.
Calculates the F3 score and latency by comparing application logs against the ground truth CSV.
"""


import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


LOG_PATH = Path("log/simulation.log")
GT_PATH = Path("data/aki.csv")


def load_ground_truth(path: Path) -> set[tuple[str, str]]:
    """
    Loads ground truth as a set of (MRN, Timestamp_String) tuples.
    Time format: YYYYMMDDHHMMSS
    """
    if not path.exists():
        print(f"Ground truth file missing: {path}")
        return set()

    df = pd.read_csv(path)

    # Normalize date columns
    date_col = next((c for c in ["date", "creatinine_date"] if c in df.columns), None)
    if not date_col:
        return set()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Convert to matching string format
    return set(zip(
        df["mrn"].astype(str),
        df[date_col].dt.strftime("%Y%m%d%H%M%S")
    ))


def parse_logs(path: Path) -> tuple[set[tuple[str, str]], list[float], float]:
    """
    Parses application JSON logs to extract predictions and latency.
    """
    predictions = set()
    latencies = list()

    if not path.exists():
        print(f"Log file missing: {path}")
        return predictions, latencies, 0.0

    timestamps = list()

    with open(path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                msg = entry.get("msg", '')
                ts_str = entry.get("ts", '')

                curr_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                timestamps.append(curr_ts)

                # Look for "PAGED: {mrn} at {event_time}"
                if msg.startswith("PAGED:"):
                    parts = msg.split()
                    mrn = parts[1]
                    event_time = parts[3]
                    predictions.add((mrn, event_time))

                    # Time since last log entry
                    if len(timestamps) > 1:
                        diff = (curr_ts - timestamps[-2]).total_seconds()
                        # filter out idle time (arbitrary 5s cutoff)
                        if diff < 5.0:
                            latencies.append(diff)
            except (json.JSONDecodeError, ValueError):
                continue

    duration = (timestamps[-1] - timestamps[0]).total_seconds() if timestamps else 0.0
    return predictions, latencies, duration


def main():
    predictions, latencies, duration = parse_logs(LOG_PATH)
    truth = load_ground_truth(GT_PATH)

    tp = len(predictions.intersection(truth))
    fp = len(predictions - truth)
    fn = len(truth - predictions)
    tn = "N/A"
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta = 3
    if (precision + recall) > 0:
        f3 = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    else:
        f3 = 0.0

    p90 = np.percentile(latencies, 90) if latencies else 0.0
    m, s = divmod(int(duration), 60)

    print("-" * 35)
    print(f"TP: {tp:<5} FP: {fp:<5}")
    print(f"FN: {fn:<5} TN: {tn:<5}")
    print("-" * 35)
    print(f"F3 Score:       {f3:.5f}")
    print(f"90% Latency:    {p90:.5f}s")
    print(f"Total Duration: {m}m {s}s")
    print("-" * 35)


if __name__ == "__main__":
    main()