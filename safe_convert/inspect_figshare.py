"""Inspect one Figshare torch archive inside the restricted container only."""
import io
import json
import tarfile

import numpy as np
import torch

ARCHIVE = "/input/battery_brand3.tar.gz"


def brief(obj, depth=0):
    if depth > 3:
        return str(type(obj))
    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype), "preview": obj.flatten()[:8].tolist()}
    if isinstance(obj, (tuple, list)):
        return [brief(x, depth + 1) for x in obj[:8]]
    if isinstance(obj, dict):
        return {str(k): brief(v, depth + 1) for k, v in list(obj.items())[:15]}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj))


with tarfile.open(ARCHIVE) as tar:
    for name in ("battery_brand3/column.pkl", "battery_brand3/data/11416.pkl"):
        raw = tar.extractfile(name).read()
        obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        print(name, json.dumps(brief(obj), ensure_ascii=False))
