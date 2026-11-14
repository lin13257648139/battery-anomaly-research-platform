"""Run ONLY inside the restricted converter container.

This module intentionally unpickles external data. Never run it on the host.
"""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def convert_one(source: Path, dest: Path) -> dict:
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda blob: torch.load(
        io.BytesIO(blob), map_location="cpu", weights_only=False
    )
    with source.open("rb") as handle:
        try:
            obj = pickle.load(handle)  # Untrusted operation: container only.
        finally:
            torch.storage._load_from_bytes = original_loader
    if type(obj) is not torch.Tensor:
        raise TypeError(f"Expected an exact torch.Tensor, got {type(obj)}")
    if obj.device.type != "cpu":
        raise ValueError("Only CPU tensors are accepted")
    arr = obj.detach().numpy()
    original_shape = list(arr.shape)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported tensor shape: {arr.shape}")
    if arr.shape[1] > 1000:
        raise ValueError(f"Unreasonably wide tensor: {arr.shape}")
    columns = {f"col_{i:03d}": pa.array(arr[:, i]) for i in range(arr.shape[1])}
    pq.write_table(pa.table(columns), dest, compression="zstd")
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else None
    return {
        "source": source.name,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_shape": original_shape,
        "parquet": dest.name,
        "rows": int(arr.shape[0]),
        "columns": int(arr.shape[1]),
        "dtype": str(arr.dtype),
        "nonfinite_count": int(arr.size - finite.sum()) if finite is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    source_root = Path("/input")
    names = sorted((p.name for p in source_root.iterdir() if p.is_dir()), key=int) if args.folder == "all" else [args.folder]
    for name in names:
        if not name.isdecimal():
            raise ValueError("folder must be numeric or all")
        folder = source_root / name
        if {p.name for p in folder.iterdir() if p.is_file()} != {"vin_1.pkl", "vin_2.pkl", "vin_3.pkl"}:
            raise ValueError(f"Unexpected source files in {name}")
        out = Path("/output") / name
        out.mkdir(parents=True, exist_ok=True)
        records = [convert_one(folder / f"vin_{i}.pkl", out / f"vin_{i}.parquet") for i in range(1, 4)]
        if len({r["rows"] for r in records}) != 1:
            raise ValueError(f"The three tensors have different row counts in {name}")
        manifest = {"source_folder": name, "records": records}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: {records[0]['rows']} rows; widths {[r['columns'] for r in records]}", flush=True)


if __name__ == "__main__":
    main()
