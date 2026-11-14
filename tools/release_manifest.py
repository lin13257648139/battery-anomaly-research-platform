"""Create or verify a content-addressed local research release snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/release_manifest.json"
FILES = [
    "converted_figshare/brand2/manifest.json",
    "features/brand2_snippets.parquet",
    "features_incremental/brand2/manifest.json",
    "models/brand2_temporal_autoencoder.pt",
    "model/build_features.py",
    "model/build_incremental_features.py",
    "model/evaluate_conditioned_brand2.py",
    "service/api.py",
    "reports/feature_quality.json",
    "reports/brand2_comparison.json",
    "reports/brand2_vehicle_scores.csv",
    "reports/brand2_failure_analysis.json",
    "reports/conditioned_brand2_results.json",
    "reports/spark_parity.json",
    "reports/stream_alert_audit.json",
]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = {name: fingerprint(ROOT / name) for name in FILES}
    if args.verify:
        recorded = json.loads(OUT.read_text(encoding="utf-8"))["files"]
        if current != recorded:
            changed = [name for name in FILES if current[name] != recorded.get(name)]
            raise SystemExit(f"Release snapshot drift: {changed}")
        print(f"Verified {len(current)} release artifacts")
    else:
        quality = json.loads((ROOT / "reports/feature_quality.json").read_text(encoding="utf-8"))
        if not quality["checks_passed"]:
            raise SystemExit("Feature quality gate has not passed")
        incremental = json.loads((ROOT / "features_incremental/brand2/manifest.json").read_text(encoding="utf-8"))
        if incremental["rows"] != quality["rows"] or not incremental["canonical_parity"]:
            raise SystemExit("Incremental feature parity has not passed")
        OUT.write_text(json.dumps({"schema_version": 1, "purpose": "local research snapshot",
                                   "files": current}, indent=2), encoding="utf-8")
        print(f"Recorded {len(current)} release artifacts")


if __name__ == "__main__":
    main()
