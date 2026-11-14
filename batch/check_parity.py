"""Compare every Spark feature against the independent pandas implementation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANDAS = ROOT / 'features' / 'brand2_snippets.parquet'
SPARK = ROOT / 'features_spark' / 'brand2'
OUT = ROOT / 'reports' / 'spark_parity.json'


def main():
    reference = pd.read_parquet(PANDAS).set_index(['car', 'snippet_id']).sort_index()
    actual = pd.read_parquet(SPARK).set_index(['car', 'snippet_id']).sort_index()
    if not reference.index.equals(actual.index):
        raise ValueError('Spark and pandas vehicle/snippet keys differ')
    if set(reference.columns) != set(actual.columns):
        raise ValueError('Spark and pandas columns differ')
    numeric = [name for name in reference if pd.api.types.is_numeric_dtype(reference[name])]
    metadata = [name for name in reference if name not in numeric]
    differing_metadata = [name for name in metadata if not reference[name].equals(actual[name])]
    errors = {}
    failures = []
    for name in numeric:
        a = reference[name].to_numpy(dtype=np.float64)
        b = actual[name].to_numpy(dtype=np.float64)
        same = np.isclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True)
        errors[name] = float(np.nanmax(np.abs(a - b)))
        if not same.all():
            failures.append({'column': name, 'differing_rows': int((~same).sum())})
    report = {'rows': len(reference), 'columns': len(reference.columns),
              'metadata_mismatch_columns': differing_metadata,
              'numeric_mismatch_columns': failures,
              'maximum_absolute_error': max(errors.values()),
              'tolerance': {'rtol': 1e-7, 'atol': 1e-9}}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if differing_metadata or failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
