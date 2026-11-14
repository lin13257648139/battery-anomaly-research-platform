"""Read-only API for the verified charging-data prototype."""
from functools import lru_cache
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'converted_figshare' / 'brand2'
REPORTS = ROOT / 'reports'
app = FastAPI(title='Battery Anomaly Research API', version='0.1.0')


@lru_cache(maxsize=1)
def scores() -> pd.DataFrame:
    return pd.read_csv(REPORTS / 'brand2_vehicle_scores.csv')


@app.get('/health')
def health():
    return {'status': 'ok', 'data_ready': (DATA / 'manifest.json').exists(),
            'scores_ready': (REPORTS / 'brand2_vehicle_scores.csv').exists()}


@app.get('/summary')
def summary():
    if not (DATA / 'manifest.json').exists():
        raise HTTPException(503, 'Converted data is not ready')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    metrics = json.loads((REPORTS / 'brand2_results.json').read_text(encoding='utf-8'))
    comparison = json.loads((REPORTS / 'brand2_comparison.json').read_text(encoding='utf-8'))
    stream = json.loads((REPORTS / 'stream_alert_audit.json').read_text(encoding='utf-8'))
    spark = json.loads((REPORTS / 'spark_batch_results.json').read_text(encoding='utf-8'))
    parity = json.loads((REPORTS / 'spark_parity.json').read_text(encoding='utf-8'))
    operating_point = json.loads((REPORTS / 'operating_point.json').read_text(encoding='utf-8'))
    return {'data': {k: v for k, v in manifest.items() if k != 'segments_per_vehicle'},
            'evaluation': metrics, 'comparison': comparison, 'historical_replay': stream,
            'spark_batch': spark, 'spark_parity': parity, 'operating_point': operating_point}


@app.get('/vehicles')
def vehicles():
    return scores().sort_values('voltage_spread', ascending=False).to_dict(orient='records')


@app.get('/research/improvements')
def research_improvements():
    """Expose verified research artifacts without loading the large feature table."""
    names = {'quality': 'feature_quality.json',
             'failure_analysis': 'brand2_failure_analysis.json',
             'conditioned_model': 'conditioned_brand2_results.json',
             'release': 'release_manifest.json'}
    missing = [name for name in names.values() if not (REPORTS / name).is_file()]
    if missing:
        raise HTTPException(503, f'Research artifacts not ready: {missing}')
    result = {key: json.loads((REPORTS / filename).read_text(encoding='utf-8'))
              for key, filename in names.items() if key != 'release'}
    release = json.loads((REPORTS / names['release']).read_text(encoding='utf-8'))
    result['release_artifact_count'] = len(release['files'])
    return result


@app.get('/vehicles/{car}/series')
def vehicle_series(car: int, max_points: int = Query(600, ge=50, le=3000)):
    path = DATA / f'car_{car}.parquet'
    if not path.is_file():
        raise HTTPException(404, 'Vehicle not found')
    cols = ['car', 'car_label', 'snippet_id', 'sample_index', 'timestamp',
            'volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp']
    parquet = pq.ParquetFile(path)
    total = parquet.metadata.num_rows
    step = max(1, total // max_points)
    chunks = []
    offset = 0
    remaining = max_points
    for batch in parquet.iter_batches(batch_size=8192, columns=cols):
        frame = batch.to_pandas()
        selected = frame.iloc[np.flatnonzero((np.arange(len(frame)) + offset) % step == 0)[:remaining]]
        if len(selected):
            chunks.append(selected)
            remaining -= len(selected)
        offset += len(frame)
        if remaining == 0:
            break
    subset = pd.concat(chunks, ignore_index=True)
    return {'car': car, 'rows_total': total, 'sampled_rows': len(subset),
            'note': 'Timestamp is the source field; a global wall-clock interpretation is not verified.',
            'points': subset.to_dict(orient='records')}
