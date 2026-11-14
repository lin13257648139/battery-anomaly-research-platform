"""Inspect one brand2 snippet inside restricted Docker."""
import io
import json
import tarfile
import numpy as np
import torch

with tarfile.open('/input/battery_brand2.tar.gz') as tar:
    for name in ('battery_brand2/column.pkl', 'battery_brand2/test/74383.pkl'):
        obj = torch.load(io.BytesIO(tar.extractfile(name).read()), map_location='cpu', weights_only=False)
        if isinstance(obj, np.ndarray):
            value = {'shape': list(obj.shape), 'dtype': str(obj.dtype)}
        elif isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], np.ndarray):
            value = {'shape': list(obj[0].shape), 'dtype': str(obj[0].dtype), 'metadata': obj[1],
                     'first_row': obj[0][0].tolist()}
        else:
            value = str(obj)[:500]
        print(name, json.dumps(value, ensure_ascii=False, default=str))
