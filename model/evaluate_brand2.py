"""Evaluate on the publisher's fixed brand2 vehicle train/test split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'features' / 'brand2_snippets.parquet'
OUT = ROOT / 'reports' / 'brand2_results.json'


def top_score(values):
    count = max(1, int(np.ceil(len(values) * 0.05)))
    return float(np.partition(values, -count)[-count:].mean())


def main():
    df = pd.read_parquet(DATA)
    cols = [c for c in df if c not in {'snippet_id', 'car', 'car_label', 'snippet_label', 'mileage', 'source_split'}]
    train_ids = set(df[df.source_split == 'train'].car)
    test_ids = set(df[df.source_split == 'test'].car)
    if train_ids & test_ids:
        raise ValueError('Train and test vehicle overlap')
    train = df[(df.source_split == 'train') & (df.car_label == 0)].copy()
    test = df[df.source_split == 'test'].copy()
    train = train.groupby('car', group_keys=True).apply(
        lambda x: x.sample(min(len(x), 1000), random_state=2026), include_groups=False
    ).reset_index(level=0).reset_index(drop=True)
    model = make_pipeline(SimpleImputer(strategy='median'), RobustScaler(),
                          IsolationForest(n_estimators=200, max_samples=256,
                                          contamination='auto', random_state=2026, n_jobs=-1))
    model.fit(train[cols])
    train['score'] = -model.decision_function(train[cols])
    normal_scores = np.sort([top_score(g.score.to_numpy()) for _, g in train.groupby('car')])
    test['score'] = -model.decision_function(test[cols])
    result_rows = []
    for car, group in test.groupby('car'):
        raw = top_score(group.score.to_numpy())
        result_rows.append({'car': int(car), 'label': int(group.car_label.iloc[0]),
                            'segments': int(len(group)), 'iforest_percentile': float(np.searchsorted(normal_scores, raw, side='right') / len(normal_scores)),
                            'voltage_spread': top_score(group.cell_volt_spread_max.to_numpy())})
    pred = pd.DataFrame(result_rows)
    result = {'source_split': 'publisher-defined vehicle train/test',
              'train_vehicles': len(train_ids), 'test_vehicles': len(test_ids),
              'test_positive_vehicles': int(pred.label.sum()), 'metrics': {}}
    for name in ('iforest_percentile', 'voltage_spread'):
        result['metrics'][name] = {'auroc': float(roc_auc_score(pred.label, pred[name])),
                                   'average_precision': float(average_precision_score(pred.label, pred[name]))}
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pred.to_csv(ROOT / 'reports' / 'brand2_vehicle_scores.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
