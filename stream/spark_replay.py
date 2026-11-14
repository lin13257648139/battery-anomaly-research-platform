"""Spark Structured Streaming replay with deduplication and durable alerts."""
import json
import sqlite3
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = Path('/app')
INPUT = ROOT / 'stream_input'
OUTPUT = ROOT / 'stream_output'
DB = OUTPUT / 'alerts.sqlite3'
CHECKPOINT = OUTPUT / 'checkpoint'
REPORT = ROOT / 'reports/stream_results.json'


def init_db():
    OUTPUT.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as db:
        db.execute('CREATE TABLE IF NOT EXISTS processed (event_id TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS pending (car INTEGER, seq INTEGER, event_id TEXT UNIQUE, score REAL, PRIMARY KEY(car,seq))')
        db.execute('CREATE TABLE IF NOT EXISTS state (car INTEGER PRIMARY KEY, last_seq INTEGER, streak INTEGER, cooldown_until INTEGER)')
        db.execute('CREATE TABLE IF NOT EXISTS alerts (event_id TEXT PRIMARY KEY, car INTEGER, seq INTEGER, score REAL, threshold REAL)')


def main():
    manifest = json.loads((ROOT / 'reports/stream_replay_manifest.json').read_text())
    threshold = float(manifest['threshold'])
    init_db()
    spark = (SparkSession.builder.appName('battery-brand2-historical-replay')
             .master('local[4]')
             .config('spark.sql.shuffle.partitions', '8')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    schema = T.StructType([
        T.StructField('event_id', T.StringType(), False),
        T.StructField('car', T.IntegerType(), False),
        T.StructField('seq', T.IntegerType(), False),
        T.StructField('event_time', T.TimestampType(), False),
        T.StructField('cell_volt_spread_max', T.DoubleType(), False),
    ])
    events = (spark.readStream.schema(schema)
              .option('maxFilesPerTrigger', 1).json(str(INPUT))
              .withWatermark('event_time', '1 hour')
              .dropDuplicates(['event_id']))

    def process(batch, batch_id):
        if batch.isEmpty():
            return
        with sqlite3.connect(DB, timeout=60) as db:
            db.execute('BEGIN IMMEDIATE')
            touched = set()
            for row in batch.toLocalIterator():
                if row.event_time is None or row.cell_volt_spread_max is None:
                    raise ValueError(f'Invalid event {row.event_id}')
                inserted = db.execute('INSERT OR IGNORE INTO processed(event_id) VALUES (?)',
                                      (row.event_id,)).rowcount
                if not inserted:
                    continue
                db.execute('INSERT INTO pending VALUES (?, ?, ?, ?)',
                           (row.car, row.seq, row.event_id, row.cell_volt_spread_max))
                touched.add(row.car)
            for car in touched:
                state = db.execute('SELECT last_seq, streak, cooldown_until FROM state WHERE car=?',
                                   (car,)).fetchone()
                last_seq, streak, cooldown = state if state else (-1, 0, -1)
                while True:
                    next_event = db.execute('SELECT event_id, score FROM pending WHERE car=? AND seq=?',
                                            (car, last_seq + 1)).fetchone()
                    if next_event is None:
                        break
                    event_id, score = next_event
                    last_seq += 1
                    if last_seq <= cooldown:
                        streak = 0
                    elif score >= threshold:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= 3:
                        db.execute('INSERT INTO alerts VALUES (?, ?, ?, ?, ?)',
                                   (event_id, car, last_seq, score, threshold))
                        cooldown = last_seq + 20
                        streak = 0
                    db.execute('DELETE FROM pending WHERE car=? AND seq=?', (car, last_seq))
                db.execute('INSERT INTO state VALUES (?, ?, ?, ?) ON CONFLICT(car) DO UPDATE SET '
                           'last_seq=excluded.last_seq, streak=excluded.streak, cooldown_until=excluded.cooldown_until',
                           (car, last_seq, streak, cooldown))
        print(f'committed microbatch {batch_id}', flush=True)

    query = (events.writeStream.foreachBatch(process)
             .option('checkpointLocation', str(CHECKPOINT))
             .trigger(availableNow=True).start())
    query.awaitTermination()
    spark.stop()
    with sqlite3.connect(DB) as db:
        processed = db.execute('SELECT COUNT(*) FROM processed').fetchone()[0]
        alerted = db.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]
        cars = db.execute('SELECT COUNT(*) FROM state').fetchone()[0]
        pending = db.execute('SELECT COUNT(*) FROM pending').fetchone()[0]
    report = {'spark_version': '3.5.9', 'source': 'synthetic-order historical file replay',
              'unique_events_processed': processed, 'injected_duplicates': manifest['injected_duplicates'],
              'alerts': alerted, 'vehicles': cars, 'pending_events': pending, 'threshold': threshold,
              'alarm_rule': '3 consecutive high snippets, 20-snippet cooldown',
              'checkpoint': str(CHECKPOINT)}
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if processed != manifest['unique_events'] or cars != manifest['test_vehicles'] or pending:
        raise ValueError('Replay completeness validation failed')


if __name__ == '__main__':
    main()
