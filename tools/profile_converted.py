"""Profile converted Parquet files. No pickle loading."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "converted"
OUT = ROOT / "reports" / "converted_profile.json"


def load_array(path: Path) -> np.ndarray:
    table = pq.read_table(path)
    return np.column_stack([col.to_numpy() for col in table.columns])


def main() -> None:
    results = []
    for folder in sorted((p for p in DATA.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        item = {"folder": folder.name, "files": {}}
        for name in ("vin_1", "vin_2", "vin_3"):
            x = load_array(folder / f"{name}.parquet")
            item["files"][name] = {
                "shape": list(x.shape),
                "minimum": x.min(axis=0).tolist(),
                "maximum": x.max(axis=0).tolist(),
                "near_zero_columns": int((np.max(np.abs(x), axis=0) < 1e-10).sum()),
                "columns_exactly_equal_to_col_002": int(sum(np.array_equal(x[:, 2], x[:, i]) for i in range(2, x.shape[1]))),
            }
        results.append(item)
    summary = {
        "folders": len(results),
        "total_rows": sum(x["files"]["vin_1"]["shape"][0] for x in results),
        "width_patterns": dict(Counter(str(tuple(x["files"][n]["shape"][1] for n in ("vin_1", "vin_2", "vin_3"))) for x in results)),
        "vin_2_duplicate_col_002_range": [min(x["files"]["vin_2"]["columns_exactly_equal_to_col_002"] for x in results), max(x["files"]["vin_2"]["columns_exactly_equal_to_col_002"] for x in results)],
        "vin_3_near_zero_columns_range": [min(x["files"]["vin_3"]["near_zero_columns"] for x in results), max(x["files"]["vin_3"]["near_zero_columns"] for x in results)],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "folders": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
