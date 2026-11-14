"""Condition-aware normal-response model for charging snippets."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'features' / 'brand3_snippets.parquet'
OUT = ROOT / 'reports' / 'context_model_results.json'
INPUTS = [f'{name}_{stat}' for name in ('current', 'soc') for stat in ('mean', 'std', 'min', 'max', 'delta')]
INPUTS += ['samples']
TARGETS = ['volt_mean', 'max_single_volt_mean', 'min_single_volt_mean', 'max_temp_mean', 'min_temp_mean']


def top_score(values):
    n = max(1, int(np.ceil(len(values) * 0.05)))
    return float(np.partition(values, -n)[-n:].mean())


def main():
    df = pd.read_parquet(DATA)
    cars = df.groupby('car', sort=True).car_label.first()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    predictions = []
    for fold, (tr, te) in enumerate(folds.split(cars.index, cars.values), 1):
        train = df[df.car.isin(cars.index[tr]) & (df.car_label == 0)].copy()
        train = train.groupby('car', group_keys=True).apply(
            lambda x: x.sample(min(len(x), 250), random_state=2026), include_groups=False
        ).reset_index(level=0).reset_index(drop=True)
        test = df[df.car.isin(cars.index[te])].copy()
        train_x = train[INPUTS].fillna(train[INPUTS].median()).to_numpy()
        train_y = train[TARGETS].to_numpy()
        test_x = test[INPUTS].fillna(train[INPUTS].median()).to_numpy()
        model = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=10,
                                    max_features=0.8, n_jobs=-1, random_state=2026)
        model.fit(train_x, train_y)
        train_residual = np.abs(train_y - model.predict(train_x))
        scale = np.maximum(np.median(train_residual, axis=0), 1e-6)
        train['score'] = np.mean(train_residual / scale, axis=1)
        calibration = np.sort(np.array([top_score(g.score.to_numpy()) for _, g in train.groupby('car')]))
        test['score'] = np.mean(np.abs(test[TARGETS].to_numpy() - model.predict(test_x)) / scale, axis=1)
        for car, group in test.groupby('car'):
            raw = top_score(group.score.to_numpy())
            predictions.append({'fold': fold, 'car': int(car), 'label': int(group.car_label.iloc[0]),
                                'score': raw, 'percentile': float(np.searchsorted(calibration, raw, side='right') / len(calibration))})
        print(f'fold {fold} complete', flush=True)
    pred = pd.DataFrame(predictions)
    result = {'task': 'vehicle anomaly ranking', 'training': 'normal vehicles only',
              'inputs': INPUTS, 'targets': TARGETS, 'vehicle_count': len(pred), 'positive_vehicles': int(pred.label.sum()),
              'oof_auroc': float(roc_auc_score(pred.label, pred.percentile)),
              'oof_average_precision': float(average_precision_score(pred.label, pred.percentile)),
              'folds': [{'fold': int(k), 'auroc': float(roc_auc_score(g.label, g.score)),
                         'average_precision': float(average_precision_score(g.label, g.score))}
                        for k, g in pred.groupby('fold')]}
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pred.to_csv(ROOT / 'reports' / 'context_vehicle_scores.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
