"""Convert labeled brand2 archive inside the restricted container."""
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SOURCE = Path('/input/battery_brand2.tar.gz')
DEST = Path('/output/brand2')
FIELDS = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
SCHEMA = pa.schema([
    ('car', pa.int32()), ('car_label', pa.int8()), ('source_split', pa.string()),
    ('snippet_id', pa.string()), ('segment_id', pa.string()), ('snippet_label', pa.string()),
    ('mileage', pa.float64()), ('sample_index', pa.int32()),
    *[(name, pa.float64()) for name in FIELDS],
])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE, 'r:gz') as tar:
        labels = {}
        for split in ('train', 'test'):
            name = f'battery_brand2/label/{split}_label.csv'
            records = csv.DictReader(io.StringIO(tar.extractfile(name).read().decode('utf-8-sig')))
            for row in records:
                car = int(row['car'])
                if car in labels:
                    raise ValueError(f'Vehicle {car} occurs in both splits')
                labels[car] = (int(row['label']), split)
        writers = {}
        pending = defaultdict(list)
        counts = Counter()
        nonfinite = 0

        def flush(car):
            if not pending[car]:
                return
            if car not in writers:
                writers[car] = pq.ParquetWriter(DEST / f'car_{car}.parquet', SCHEMA, compression='zstd')
            writers[car].write_table(pa.Table.from_pylist(pending[car], schema=SCHEMA))
            pending[car].clear()

        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.pkl'):
                    continue
                parts = Path(member.name).parts
                if len(parts) != 3 or parts[1] not in ('train', 'test'):
                    continue
                if member.size > 10_000_000:
                    raise ValueError(f'Unusually large snippet: {member.name}')
                obj = torch.load(io.BytesIO(tar.extractfile(member).read()), map_location='cpu', weights_only=False)
                if not isinstance(obj, (tuple, list)) or len(obj) != 2:
                    raise TypeError(f'Unexpected object: {member.name}')
                array, meta = obj
                if type(array) is not np.ndarray or array.ndim != 2 or array.shape[1] != 8:
                    raise ValueError(f'Unexpected shape: {member.name}')
                if not np.isfinite(array).all():
                    nonfinite += 1
                    continue
                car = int(meta['car'])
                if car not in labels or labels[car][1] != parts[1]:
                    raise ValueError(f'Label or split mismatch: {member.name}')
                label, split = labels[car]
                snippet_id = Path(member.name).stem
                for i, values in enumerate(array):
                    row = {'car': car, 'car_label': label, 'source_split': split,
                           'snippet_id': snippet_id, 'segment_id': str(meta['charge_segment']),
                           'snippet_label': str(meta['label']), 'mileage': float(meta['mileage']),
                           'sample_index': i}
                    row.update({field: float(value) for field, value in zip(FIELDS, values)})
                    pending[car].append(row)
                counts[car] += 1
                if len(pending[car]) >= 8192:
                    flush(car)
                if sum(counts.values()) % 20000 == 0:
                    print(f"processed {sum(counts.values())} snippets", flush=True)
            for car in list(pending):
                flush(car)
        finally:
            for writer in writers.values():
                writer.close()
    summary = {'source': 'https://doi.org/10.6084/m9.figshare.23659323.v1',
               'brand': 'brand2', 'vehicle_count': len(counts), 'segments': sum(counts.values()),
               'label_counts': dict(Counter(label for label, _ in labels.values())),
               'split_counts': dict(Counter(split for _, split in labels.values())),
               'nonfinite_segments_skipped': nonfinite, 'fields': FIELDS}
    (DEST / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
