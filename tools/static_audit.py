"""Inspect GIS pickle opcodes without deserializing or executing them."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pickletools

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "GIS" / "GIS"
OUT = ROOT / "reports" / "static_audit.json"
EXPECTED_FILES = {"vin_1.pkl", "vin_2.pkl", "vin_3.pkl"}


def inspect(path: Path) -> dict:
    blob = path.read_bytes()
    ops = list(pickletools.genops(blob))
    strings = [arg for op, arg, _ in ops if op.name in {"SHORT_BINUNICODE", "BINUNICODE"}]
    numbers = [arg for op, arg, _ in ops if op.name in {"BININT1", "BININT2", "BININT"}]
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": len(blob),
        "sha256": sha256(blob).hexdigest(),
        "protocol": ops[0][1] if ops and ops[0][0].name == "PROTO" else None,
        "global_module_strings": [x for x in strings if x in {"torch._utils", "torch.storage", "collections"}],
        "global_function_strings": [x for x in strings if x in {"_rebuild_tensor_v2", "_load_from_bytes", "OrderedDict"}],
        "integer_prefix": numbers[:9],
        "opcode_counts": dict(Counter(op.name for op, _, _ in ops)),
    }


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"Missing data directory: {DATA}")
    dirs = sorted((p for p in DATA.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    for folder in dirs:
        if {p.name for p in folder.iterdir() if p.is_file()} != EXPECTED_FILES:
            raise SystemExit(f"Unexpected files in {folder}")
    files = [folder / name for folder in dirs for name in sorted(EXPECTED_FILES)]
    report = {
        "source": "https://doi.org/10.5281/zenodo.10656500",
        "directory_count": len(dirs),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "note": "Static pickle opcode inspection only; no file was deserialized.",
        "files": [inspect(p) for p in files],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited {len(files)} files across {len(dirs)} directories; {report['total_bytes']} bytes")
    print(OUT)


if __name__ == "__main__":
    main()
