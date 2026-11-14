"""Record the exact local Python stack used for model experiments."""
import json
import platform
from pathlib import Path

import numpy
import pandas
import pyarrow
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
report = {'python': platform.python_version(), 'platform': platform.platform(),
          'numpy': numpy.__version__, 'pandas': pandas.__version__,
          'pyarrow': pyarrow.__version__, 'scikit_learn': sklearn.__version__,
          'torch': torch.__version__, 'cuda_available_in_this_python': torch.cuda.is_available(),
          'torch_threads': torch.get_num_threads()}
(ROOT / 'reports/environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
