"""Audit simulated replay completeness and vehicle alert outcomes."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'stream_output/alerts.sqlite3'
MANIFEST = ROOT / 'reports/stream_replay_manifest.json'
OUT = ROOT / 'reports/stream_alert_audit.json'


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    labels = (pd.read_parquet(ROOT / 'features/brand2_snippets.parquet',
                              columns=['car', 'car_label', 'source_split'])
              .query("source_split == 'test'")
              .groupby('car').car_label.first())
    with sqlite3.connect(DB) as db:
        processed = db.execute('SELECT COUNT(*) FROM processed').fetchone()[0]
        pending = db.execute('SELECT COUNT(*) FROM pending').fetchone()[0]
        alerts = pd.read_sql_query('SELECT car, seq, score FROM alerts', db)
        final = pd.read_sql_query('SELECT car, last_seq FROM state', db).set_index('car')
    alerted = set(alerts.car.unique()) if len(alerts) else set()
    positive = set(labels[labels == 1].index)
    negative = set(labels[labels == 0].index)
    report = {'unique_events_expected': manifest['unique_events'],
              'unique_events_processed': processed,
              'injected_duplicates': manifest['injected_duplicates'],
              'pending_events': pending,
              'test_vehicles': len(labels), 'alert_events': len(alerts),
              'alerted_positive_vehicles': len(alerted & positive),
              'alerted_normal_vehicles': len(alerted & negative),
              'positive_vehicles': len(positive), 'normal_vehicles': len(negative),
              'alerted_cars': sorted(int(x) for x in alerted),
              'caveat': 'Synthetic ordering; no real-world detection latency or lead time.'}
    if processed != manifest['unique_events'] or pending or len(final) != len(labels):
        raise ValueError('Replay completeness check failed')
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
