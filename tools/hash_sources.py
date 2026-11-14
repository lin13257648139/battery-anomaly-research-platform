"""Record immutable SHA-256 identifiers for downloaded source data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/source_hashes.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    paths = sorted((ROOT / 'GIS/GIS').glob('*/vin_*.pkl'))
    paths.extend([ROOT / 'external_data/figshare_ev/battery_brand2.tar.gz',
                  ROOT / 'external_data/figshare_ev/battery_brand3.tar.gz'])
    if len(paths) != 125 or not all(path.is_file() for path in paths):
        raise ValueError('Expected 123 GIS pickles and two Figshare archives')
    records = []
    for i, path in enumerate(paths, 1):
        records.append({'path': path.relative_to(ROOT).as_posix(),
                        'bytes': path.stat().st_size, 'sha256': sha256(path)})
        if i % 20 == 0:
            print(f'hashed {i}/{len(paths)}', flush=True)
    report = {'algorithm': 'SHA-256', 'file_count': len(records),
              'total_bytes': sum(row['bytes'] for row in records), 'files': records}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'file_count': report['file_count'],
                      'total_bytes': report['total_bytes']}, indent=2))


if __name__ == '__main__':
    main()
