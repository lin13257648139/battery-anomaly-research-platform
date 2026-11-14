"""Build one feature vector per charging snippet from safe Parquet data."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp']


def extract_vehicle(path: Path) -> pd.DataFrame:
    """Canonical snippet feature calculation shared by full and incremental jobs."""
    df = pq.read_table(path).to_pandas()
    df['cell_volt_spread'] = df['max_single_volt'] - df['min_single_volt']
    df['temp_spread'] = df['max_temp'] - df['min_temp']
    fields = SIGNALS + ['cell_volt_spread', 'temp_spread']
    g = df.groupby('snippet_id', sort=False, observed=True)
    stat = g[fields].agg(['mean', 'std', 'min', 'max'])
    stat.columns = [f'{field}_{measure}' for field, measure in stat.columns]
    first = g[fields].first()
    last = g[fields].last()
    delta = last - first
    delta.columns = [f'{name}_delta' for name in delta.columns]
    meta_cols = ['car', 'car_label', 'snippet_label', 'mileage']
    if 'source_split' in df:
        meta_cols.append('source_split')
    meta = g[meta_cols].first()
    length = g.size().rename('samples')
    if not length.eq(128).all():
        raise ValueError(f'Snippet length or ID collision in {path}')
    return pd.concat([meta, length, stat, delta], axis=1).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--brand', choices=['brand2', 'brand3'], default='brand3')
    args = parser.parse_args()
    source = ROOT / 'converted_figshare' / args.brand
    out = ROOT / 'features' / f'{args.brand}_snippets.parquet'
    frames = []
    for path in sorted(source.glob('car_*.parquet')):
        frames.append(extract_vehicle(path))
    result = pd.concat(frames, ignore_index=True)
    if result[['car', 'snippet_id']].duplicated().any():
        raise ValueError('Duplicate snippet IDs within a vehicle')
    converted_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if len(result) != converted_manifest['segments']:
        raise ValueError('Feature count differs from converted snippet count')
    feature_cols = [c for c in result if c not in {'snippet_id', 'car', 'car_label', 'snippet_label', 'mileage', 'source_split'}]
    result[feature_cols] = result[feature_cols].replace([np.inf, -np.inf], np.nan)
    missing = result[feature_cols].isna().sum().sum()
    out.parent.mkdir(exist_ok=True)
    result.to_parquet(out, index=False)
    manifest = {'source': str(source), 'brand': args.brand, 'snippets': int(len(result)),
                'cars': int(result.car.nunique()), 'positive_cars': int(result.groupby('car').car_label.first().sum()),
                'features': feature_cols, 'missing_feature_cells': int(missing)}
    (out.parent / f'{args.brand}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'features'}, indent=2))


if __name__ == '__main__':
    main()
