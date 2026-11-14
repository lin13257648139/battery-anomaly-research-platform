"""Audit the canonical brand-2 feature table without changing model results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "features/brand2_snippets.parquet"
MANIFEST = ROOT / "converted_figshare/brand2/manifest.json"
OUT = ROOT / "reports/feature_quality.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    table = pq.ParquetFile(SOURCE)
    fields = table.schema_arrow.names
    key = ["car", "snippet_id"]
    metadata = ["car_label", "snippet_label", "source_split", "samples"]
    relevant = key + [name for name in metadata if name in fields]
    relevant += [name for name in fields if name not in relevant]
    frame = pd.read_parquet(SOURCE, columns=relevant)
    numeric = frame.select_dtypes(include="number")
    invalid_numeric = {name: int((~np.isfinite(numeric[name].to_numpy(dtype=float))).sum())
                       for name in numeric.columns}
    label_conflicts = int(frame.groupby("car").car_label.nunique().gt(1).sum())
    split_conflicts = int(frame.groupby("car").source_split.nunique().gt(1).sum())
    report = {
        "dataset": "figshare_brand2_snippet_features",
        "feature_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(SOURCE),
        "bytes": SOURCE.stat().st_size,
        "schema": {field.name: str(field.type) for field in table.schema_arrow},
        "rows": int(len(frame)),
        "vehicles": int(frame.car.nunique()),
        "duplicate_vehicle_snippet_keys": int(frame.duplicated(key).sum()),
        "non_128_sample_snippets": int(frame.samples.ne(128).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "nonfinite_numeric_cells": int(sum(invalid_numeric.values())),
        "nonfinite_numeric_by_column": {k: v for k, v in invalid_numeric.items() if v},
        "vehicle_label_conflicts": label_conflicts,
        "vehicle_split_conflicts": split_conflicts,
        "source_manifest_segments": int(source_manifest["segments"]),
        "checks_passed": bool(len(frame) == source_manifest["segments"]
                              and not frame.duplicated(key).any()
                              and frame.samples.eq(128).all()
                              and not frame.isna().any().any()
                              and not any(invalid_numeric.values())
                              and label_conflicts == 0 and split_conflicts == 0),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "vehicles", "sha256", "checks_passed")}, indent=2))
    if not report["checks_passed"]:
        raise SystemExit("Feature quality audit failed")


if __name__ == "__main__":
    main()
