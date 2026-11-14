"""Paired stratified vehicle bootstrap for the fixed brand2 test set."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports' / 'brand2_vehicle_scores.csv'
DEEP = ROOT / 'reports' / 'temporal_autoencoder_vehicle_scores.csv'
OUT = ROOT / 'reports' / 'brand2_comparison.json'


def metrics(labels, scores):
    return {'auroc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores))}


def main():
    baseline = pd.read_csv(BASE)
    deep = pd.read_csv(DEEP)
    data = baseline.merge(deep[['car', 'label', 'score']], on=['car', 'label'],
                          validate='one_to_one')
    if len(data) != 18 or int(data.label.sum()) != 7:
        raise ValueError('Test vehicle or label mismatch')
    columns = ['voltage_spread', 'iforest_percentile', 'score']
    names = ['cell_voltage_spread', 'isolation_forest', 'temporal_autoencoder']
    positive = np.flatnonzero(data.label.to_numpy() == 1)
    negative = np.flatnonzero(data.label.to_numpy() == 0)
    rng = np.random.default_rng(2026)
    samples = {name: {'auroc': [], 'average_precision': []} for name in names}
    for _ in range(2000):
        sample = np.concatenate([rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)])
        for name, column in zip(names, columns):
            result = metrics(data.label.to_numpy()[sample], data[column].to_numpy()[sample])
            for metric, value in result.items():
                samples[name][metric].append(value)
    report = {'protocol': 'publisher fixed vehicle test split; stratified paired 2000-resample bootstrap',
              'test_vehicles': len(data), 'positive_vehicles': len(positive), 'models': {}}
    for name, column in zip(names, columns):
        point = metrics(data.label, data[column])
        report['models'][name] = {metric: {'point': point[metric],
                                           'ci95': [float(x) for x in np.quantile(samples[name][metric], [.025, .975])]}
                                  for metric in point}
    # Differences are paired on the same resampled vehicles. They are exploratory
    # because this public test set is small and models were developed on related data.
    report['difference_vs_physical'] = {}
    for name in names[1:]:
        report['difference_vs_physical'][name] = {}
        for metric in ('auroc', 'average_precision'):
            delta = np.array(samples[name][metric]) - np.array(samples['cell_voltage_spread'][metric])
            report['difference_vs_physical'][name][metric] = {
                'point': report['models'][name][metric]['point'] - report['models']['cell_voltage_spread'][metric]['point'],
                'ci95': [float(x) for x in np.quantile(delta, [.025, .975])]}
    # This criterion was selected for data quality, not to maximize a metric.
    # The smallest test vehicle has only five snippets.
    enough = data[data.segments >= 20]
    report['sensitivity_min_20_snippets'] = {
        'test_vehicles': len(enough), 'positive_vehicles': int(enough.label.sum()),
        'excluded_vehicles': data.loc[data.segments < 20, 'car'].astype(int).tolist(),
        'metrics': {name: metrics(enough.label, enough[column])
                    for name, column in zip(names, columns)}}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
