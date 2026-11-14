"""Convert Figshare torch archives to vehicle-partitioned Parquet inside Docker."""
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

SOURCE = Path('/input/battery_brand3.tar.gz')
DEST = Path('/output/brand3')
FIELDS = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
SCHEMA = pa.schema([
    ('car', pa.int32()), ('car_label', pa.int8()), ('snippet_id', pa.string()), ('segment_id', pa.string()),
    ('snippet_label', pa.string()), ('mileage', pa.float64()), ('sample_index', pa.int32()),
    *[(name, pa.float64()) for name in FIELDS],
])


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    writers = {}
    pending = defaultdict(list)
    counts = Counter()
    snippet_labels = Counter()
    bad = []

    def flush(car: int) -> None:
        rows = pending[car]
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        if car not in writers:
            writers[car] = pq.ParquetWriter(DEST / f'car_{car}.parquet', SCHEMA, compression='zstd')
        writers[car].write_table(table)
        pending[car].clear()

    try:
        with tarfile.open(SOURCE, mode='r|gz') as tar:
            labels = None
            for member in tar:
                if not member.isfile():
                    continue
                if member.name == 'battery_brand3/label/all_label.csv':
                    text = tar.extractfile(member).read().decode('utf-8-sig')
                    labels = {int(r['car']): int(r['label']) for r in csv.DictReader(io.StringIO(text))}
                    continue
                if not member.name.startswith('battery_brand3/data/') or not member.name.endswith('.pkl'):
                    continue
                if labels is None:
                    raise ValueError('Vehicle labels were not found before data files')
                if member.size > 10_000_000:
                    raise ValueError(f'Unusually large snippet: {member.name}')
                raw = tar.extractfile(member).read()
                obj = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                if not isinstance(obj, (list, tuple)) or len(obj) != 2:
                    raise TypeError(f'Unexpected object: {member.name}')
                array, metadata = obj
                if type(array) is not np.ndarray or array.ndim != 2 or array.shape[1] != 8:
                    raise ValueError(f'Unexpected array shape: {member.name}')
                if not np.isfinite(array).all():
                    bad.append(member.name)
                    continue
                car = int(metadata['car'])
                if car not in labels:
                    raise ValueError(f'Missing vehicle label for car {car}')
                snippet_id = Path(member.name).stem
                segment = str(metadata['charge_segment'])
                snippet_label = str(metadata['label'])
                mileage = float(metadata['mileage'])
                snippet_labels[snippet_label] += 1
                for i, values in enumerate(array):
                    row = {'car': car, 'car_label': labels[car], 'snippet_id': snippet_id, 'segment_id': segment,
                           'snippet_label': snippet_label, 'mileage': mileage, 'sample_index': i}
                    row.update({field: float(value) for field, value in zip(FIELDS, values)})
                    pending[car].append(row)
                counts[car] += 1
                if len(pending[car]) >= 8192:
                    flush(car)
            if labels is None:
                raise ValueError('Missing vehicle label CSV')
        for car in list(pending):
            flush(car)
    finally:
        for writer in writers.values():
            writer.close()
    summary = {'vehicles_with_data': len(counts), 'vehicle_label_counts': dict(Counter(labels.values())),
               'segments': sum(counts.values()), 'segments_per_vehicle': dict(counts),
               'snippet_label_counts': dict(snippet_labels), 'nonfinite_segments_skipped': bad,
               'fields': FIELDS, 'source': 'https://doi.org/10.6084/m9.figshare.23659323.v1'}
    (DEST / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('segments_per_vehicle', 'nonfinite_segments_skipped')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
