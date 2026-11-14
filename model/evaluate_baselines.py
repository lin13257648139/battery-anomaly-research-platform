"""Vehicle-stratified 5-fold evaluation, with no vehicle shared across train/test."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'features' / 'brand3_snippets.parquet'
OUT = ROOT / 'reports' / 'baseline_results.json'


def top_score(values: np.ndarray) -> float:
    count = max(1, int(np.ceil(len(values) * 0.05)))
    return float(np.partition(values, -count)[-count:].mean())


def main() -> None:
    df = pd.read_parquet(DATA)
    features = [c for c in df if c not in {'snippet_id', 'car', 'car_label', 'snippet_label', 'mileage'}]
    cars = df.groupby('car', sort=True).car_label.first()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    predictions = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(cars.index, cars.values), 1):
        train_cars = cars.index[train_idx]
        test_cars = cars.index[test_idx]
        train = df[df.car.isin(train_cars) & (df.car_label == 0)].copy()
        test = df[df.car.isin(test_cars)].copy()
        # Cap each normal vehicle to avoid letting long histories dominate.
        train = train.groupby('car', group_keys=True).apply(
            lambda x: x.sample(min(len(x), 250), random_state=2026), include_groups=False
        ).reset_index(level=0).reset_index(drop=True)
        pipeline = make_pipeline(SimpleImputer(strategy='median'), RobustScaler(),
                                 IsolationForest(n_estimators=200, max_samples=256,
                                                 contamination='auto', random_state=2026, n_jobs=-1))
        pipeline.fit(train[features])
        train = train.copy()
        train['iforest_score'] = -pipeline.decision_function(train[features])
        calibration = np.sort(np.array([top_score(g.iforest_score.to_numpy()) for _, g in train.groupby('car')]))
        test = test.copy()
        test['iforest_score'] = -pipeline.decision_function(test[features])
        test['voltage_spread_score'] = test['cell_volt_spread_max']
        for car, group in test.groupby('car'):
            raw_iforest = top_score(group.iforest_score.to_numpy())
            predictions.append({'fold': fold, 'car': int(car), 'label': int(group.car_label.iloc[0]),
                                'segments': int(len(group)),
                                'iforest': raw_iforest,
                                'iforest_percentile': float(np.searchsorted(calibration, raw_iforest, side='right') / len(calibration)),
                                'voltage_spread': top_score(group.voltage_spread_score.to_numpy())})
    pred = pd.DataFrame(predictions)
    result = {'unit': 'vehicle', 'folds': 5, 'vehicles': int(len(pred)),
              'positives': int(pred.label.sum()), 'split': 'StratifiedKFold on vehicle ID; training uses normal vehicles only',
              'aggregation': 'mean of highest 5% snippet scores', 'metrics': {}}
    for name in ('iforest_percentile', 'voltage_spread'):
        result['metrics'][name] = {
            'oof_auroc': float(roc_auc_score(pred.label, pred[name])),
            'oof_average_precision': float(average_precision_score(pred.label, pred[name])),
            'folds': [
                {'fold': int(fold), 'auroc': float(roc_auc_score(part.label, part[name])),
                 'average_precision': float(average_precision_score(part.label, part[name]))}
                for fold, part in pred.groupby('fold')
            ],
        }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pred.to_csv(OUT.parent / 'baseline_vehicle_scores.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
