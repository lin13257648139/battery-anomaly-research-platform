"""Per-vehicle, content-addressed feature materialization with restart safety."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from build_features import ROOT, extract_vehicle

SOURCE = ROOT / "converted_figshare/brand2"
DEST = ROOT / "features_incremental/brand2"
MANIFEST = DEST / "manifest.json"
CANONICAL = ROOT / "features/brand2_snippets.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob("car_*.parquet"))
    expected = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if len(paths) != expected["vehicle_count"]:
        raise ValueError("Source file count differs from converted manifest")
    code_hash = sha256(ROOT / "model/build_features.py")
    prior = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    previous = prior.get("vehicles", {})
    current_names = {p.stem for p in paths}
    if set(previous) - current_names:
        raise ValueError("Source vehicle disappeared; inspect before publishing a new snapshot")
    state = {"schema_version": 1, "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
             "feature_code_sha256": code_hash, "vehicles": dict(previous)}
    changed = 0
    for path in paths:
        name = path.stem
        target = DEST / f"{name}.parquet"
        source_hash = sha256(path)
        old = state["vehicles"].get(name, {})
        if (old.get("source_sha256") == source_hash and
                old.get("feature_code_sha256") == code_hash and target.exists() and
                old.get("output_sha256") == sha256(target)):
            continue
        features = extract_vehicle(path)
        if features[["car", "snippet_id"]].duplicated().any():
            raise ValueError(f"Duplicate feature key in {name}")
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}-", suffix=".parquet", dir=DEST)
        os.close(fd)
        try:
            features.to_parquet(temp_name, index=False)
            output_hash = sha256(Path(temp_name))
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        state["vehicles"][name] = {"source_sha256": source_hash,
                                    "feature_code_sha256": code_hash,
                                    "output_sha256": output_hash,
                                    "rows": int(len(features))}
        atomic_json(MANIFEST, state)
        changed += 1
        print(f"updated {name}: {len(features)} snippets", flush=True)
    total = sum(v["rows"] for v in state["vehicles"].values())
    if total != expected["segments"]:
        raise ValueError(f"Feature total {total} differs from source manifest {expected['segments']}")
    if CANONICAL.exists():
        reference = pd.read_parquet(CANONICAL).set_index(["car", "snippet_id"]).sort_index()
        output = pd.concat((pd.read_parquet(DEST / f"{name}.parquet") for name in sorted(state["vehicles"])),
                           ignore_index=True).set_index(["car", "snippet_id"]).sort_index()
        pd.testing.assert_frame_equal(reference, output, check_exact=False, rtol=1e-12, atol=1e-12)
    state["rows"] = total
    state["vehicle_count"] = len(state["vehicles"])
    state["canonical_parity"] = CANONICAL.exists()
    atomic_json(MANIFEST, state)
    print(json.dumps({"updated_vehicles": changed, "vehicles": state["vehicle_count"],
                      "rows": total, "canonical_parity": state["canonical_parity"]}))


if __name__ == "__main__":
    main()
