"""Distributed-equivalent snippet features from safe Parquet (Spark 3.5.9)."""
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

ROOT = Path('/app')
SIGNALS = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp']


def main():
    spark = (SparkSession.builder.appName('battery-brand2-features')
             .master('local[4]')
             .config('spark.sql.shuffle.partitions', '16')
             .config('spark.driver.memory', '3g')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    begin = time.monotonic()
    rows = spark.read.parquet(str(ROOT / 'converted_figshare/brand2/car_*.parquet'))
    rows = (rows.withColumn('cell_volt_spread', F.col('max_single_volt') - F.col('min_single_volt'))
            .withColumn('temp_spread', F.col('max_temp') - F.col('min_temp')))
    fields = SIGNALS + ['cell_volt_spread', 'temp_spread']
    aggregations = [F.first(c, ignorenulls=False).alias(c) for c in
                    ['car_label', 'source_split', 'snippet_label', 'mileage']]
    aggregations.append(F.count('*').alias('samples'))
    for field in fields:
        aggregations.extend([
            F.mean(field).alias(f'{field}_mean'),
            F.stddev_samp(field).alias(f'{field}_std'),
            F.min(field).alias(f'{field}_min'),
            F.max(field).alias(f'{field}_max'),
            (F.max_by(F.col(field), F.col('sample_index')) -
             F.min_by(F.col(field), F.col('sample_index'))).alias(f'{field}_delta'),
        ])
    features = rows.groupBy('car', 'snippet_id').agg(*aggregations)
    invalid = features.filter(F.col('samples') != 128).count()
    if invalid:
        raise ValueError(f'{invalid} groups do not have 128 samples')
    count = features.count()
    expected = json.loads((ROOT / 'converted_figshare/brand2/manifest.json').read_text())['segments']
    if count != expected:
        raise ValueError(f'{count} features != {expected} converted snippets')
    output = ROOT / 'features_spark/brand2'
    features.coalesce(4).write.mode('overwrite').parquet(str(output))
    duration = time.monotonic() - begin
    report = {'spark_version': spark.version, 'mode': 'local[4]',
              'input_rows': expected * 128, 'output_snippets': count,
              'invalid_length_groups': invalid, 'elapsed_seconds': duration}
    (ROOT / 'reports/spark_batch_results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    spark.stop()


if __name__ == '__main__':
    main()
