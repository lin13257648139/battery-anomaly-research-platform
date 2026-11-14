"""Prepare deterministic synthetic-order test-vehicle replay files.

Snippet IDs are source identifiers, not verified wall-clock order. The resulting
event_time is synthetic and cannot support a lead-time claim.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'features/brand2_snippets.parquet'
DEST = ROOT / 'stream_input'
MANIFEST = ROOT / 'reports/stream_replay_manifest.json'


def main():
    data = pd.read_parquet(SOURCE, columns=['car', 'snippet_id', 'source_split',
                                            'car_label', 'cell_volt_spread_max'])
    normal_train = data[(data.source_split == 'train') & (data.car_label == 0)]
    threshold = float(np.quantile(normal_train.cell_volt_spread_max, .99))
    test = data[data.source_split == 'test'].copy()
    test['numeric_id'] = pd.to_numeric(test.snippet_id, errors='raise')
    test = test.sort_values(['car', 'numeric_id']).reset_index(drop=True)
    test['seq'] = test.groupby('car').cumcount()
    if test[['car', 'snippet_id']].duplicated().any():
        raise ValueError('Duplicate source test identifiers')
    DEST.mkdir(exist_ok=True)
    for path in DEST.glob('part_*.json'):
        path.unlink()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    duplicate_count = 0
    output_rows = []
    for i, row in enumerate(test.itertuples(index=False)):
        event = {'event_id': f'{int(row.car)}:{row.snippet_id}', 'car': int(row.car),
                 'seq': int(row.seq), 'event_time': (start + timedelta(seconds=i)).isoformat(),
                 'cell_volt_spread_max': float(row.cell_volt_spread_max)}
        output_rows.append(event)
        if i % 100 == 0:
            output_rows.append(event.copy())
            duplicate_count += 1
    for part, begin in enumerate(range(0, len(output_rows), 5000)):
        path = DEST / f'part_{part:04d}.json'
        with path.open('w', encoding='utf-8') as stream:
            for event in output_rows[begin:begin + 5000]:
                stream.write(json.dumps(event, separators=(',', ':')) + '\n')
    manifest = {'source': 'brand2 publisher-defined test vehicles',
                'order': 'numeric snippet identifier; synthetic, not verified chronology',
                'unique_events': len(test), 'injected_duplicates': duplicate_count,
                'input_events': len(output_rows), 'files': len(list(DEST.glob('part_*.json'))),
                'threshold_source': '99th percentile of normal training snippets',
                'threshold': threshold, 'test_vehicles': int(test.car.nunique())}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
