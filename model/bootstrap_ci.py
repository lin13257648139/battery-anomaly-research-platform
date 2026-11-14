"""Stratified vehicle bootstrap intervals for out-of-fold rankings."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports' / 'baseline_vehicle_scores.csv'
OUT = ROOT / 'reports' / 'baseline_intervals.json'


def main() -> None:
    df = pd.read_csv(DATA)
    normal = df[df.label == 0]
    abnormal = df[df.label == 1]
    rng = np.random.default_rng(2026)
    result = {'method': 'stratified bootstrap over out-of-fold vehicle scores',
              'resamples': 2000, 'normal_vehicles': len(normal), 'abnormal_vehicles': len(abnormal),
              'metrics': {}}
    for name in ('iforest_percentile', 'voltage_spread'):
        aurocs, aps = [], []
        for _ in range(2000):
            sample = pd.concat([
                normal.iloc[rng.integers(len(normal), size=len(normal))],
                abnormal.iloc[rng.integers(len(abnormal), size=len(abnormal))],
            ])
            aurocs.append(roc_auc_score(sample.label, sample[name]))
            aps.append(average_precision_score(sample.label, sample[name]))
        result['metrics'][name] = {
            'auroc_95pct_interval': np.quantile(aurocs, [0.025, 0.975]).tolist(),
            'average_precision_95pct_interval': np.quantile(aps, [0.025, 0.975]).tolist(),
        }
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
