"""Train-normal calibrated terminal vehicle screening operating point."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def top_five_percent(values):
    count = max(1, int(np.ceil(len(values) * .05)))
    return float(np.partition(values, -count)[-count:].mean())


def main():
    data = pd.read_parquet(ROOT / 'features/brand2_snippets.parquet',
                           columns=['car', 'car_label', 'source_split', 'cell_volt_spread_max'])
    normal = data[(data.source_split == 'train') & (data.car_label == 0)]
    calibration = normal.groupby('car').cell_volt_spread_max.apply(
        lambda x: top_five_percent(x.to_numpy())).to_numpy()
    threshold = float(np.quantile(calibration, .95))
    test = pd.read_csv(ROOT / 'reports/brand2_vehicle_scores.csv')
    test['screen_positive'] = test.voltage_spread >= threshold
    tp = int(((test.label == 1) & test.screen_positive).sum())
    fp = int(((test.label == 0) & test.screen_positive).sum())
    fn = int(((test.label == 1) & ~test.screen_positive).sum())
    tn = int(((test.label == 0) & ~test.screen_positive).sum())
    result = {'protocol': 'terminal vehicle-level screening, exploratory after initial test review',
              'calibration': '95th percentile of normal training-vehicle top-5% snippet scores',
              'train_normal_vehicles': len(calibration), 'threshold': threshold,
              'test_vehicles': len(test), 'true_positive': tp, 'false_positive': fp,
              'false_negative': fn, 'true_negative': tn,
              'limitation': 'Requires all available snippets per vehicle; not a validated online alert rule.'}
    (ROOT / 'reports/operating_point.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
