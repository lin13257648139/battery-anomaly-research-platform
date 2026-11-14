"""Normal-only temporal reconstruction baseline on the publisher's brand2 split.

The held-out test vehicles are read only after training and validation have ended.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'converted_figshare' / 'brand2'
REPORT = ROOT / 'reports' / 'temporal_autoencoder_results.json'
SIGNALS = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp']
SEED = 2026


class Autoencoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def read_car(path, limit=None):
    cols = ['snippet_id', 'sample_index', 'car_label', 'source_split'] + SIGNALS
    frame = pd.read_parquet(path, columns=cols)
    car_label = int(frame.car_label.iloc[0])
    split = str(frame.source_split.iloc[0])
    if frame.car_label.nunique() != 1 or frame.source_split.nunique() != 1:
        raise ValueError(f'Inconsistent vehicle metadata: {path}')
    groups = frame.groupby('snippet_id', sort=False)
    ids = list(groups.indices)
    if limit is not None and len(ids) > limit:
        ids = sorted(random.Random(SEED + int(path.stem.split('_')[1])).sample(ids, limit))
    arrays = []
    for key in ids:
        group = groups.get_group(key).sort_values('sample_index')
        if len(group) != 128 or not np.array_equal(group.sample_index.to_numpy(), np.arange(128)):
            raise ValueError(f'Unexpected sample index for {path}: {key}')
        x = group[SIGNALS].to_numpy(dtype=np.float32)
        spread = (x[:, 3] - x[:, 4])[:, None]
        temp_spread = (x[:, 5] - x[:, 6])[:, None]
        arrays.append(np.concatenate([x, spread, temp_spread], axis=1))
    return np.stack(arrays), car_label, split


def car_score(scores):
    n = max(1, int(np.ceil(len(scores) * .05)))
    return float(np.partition(scores, -n)[-n:].mean())


def score_arrays(model, arrays, center, scale):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(arrays), 256):
            x = torch.from_numpy(((arrays[start:start + 256] - center) / scale).transpose(0, 2, 1).copy())
            reconstruction = model(x)
            scores.extend((reconstruction - x).square().mean(dim=(1, 2)).numpy())
    return np.asarray(scores)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    paths = sorted(SOURCE.glob('car_*.parquet'))
    if len(paths) != 49:
        raise ValueError(f'Expected 49 brand2 vehicles with snippets, found {len(paths)}')
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['vehicle_count'] != 49:
        raise ValueError('Brand2 manifest vehicle count mismatch')

    # Read the publisher's training split first. A per-car cap prevents large cars
    # from dominating optimization; validation uses entirely separate normal cars.
    normal = []
    test_paths = []
    for path in paths:
        # Parquet metadata columns are cheap to read and do not touch test signals.
        info = pd.read_parquet(path, columns=['car_label', 'source_split']).iloc[0]
        if info.source_split == 'test':
            test_paths.append(path)
        elif info.source_split == 'train' and int(info.car_label) == 0:
            normal.append(path)
    if len(test_paths) != 18 or len(normal) != 22:
        raise ValueError(f'Unexpected split: normal train={len(normal)}, test={len(test_paths)}')
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(normal))
    val_paths = [normal[i] for i in order[:5]]
    fit_paths = [normal[i] for i in order[5:]]
    fit = np.concatenate([read_car(path, limit=400)[0] for path in fit_paths])
    val = np.concatenate([read_car(path, limit=400)[0] for path in val_paths])
    center = np.median(fit, axis=(0, 1)).astype(np.float32)
    q25, q75 = np.percentile(fit, [25, 75], axis=(0, 1))
    scale = np.maximum((q75 - q25).astype(np.float32), 1e-3)
    fit_x = torch.from_numpy(((fit - center) / scale).transpose(0, 2, 1).copy())
    val_x = torch.from_numpy(((val - center) / scale).transpose(0, 2, 1).copy())
    loader = DataLoader(TensorDataset(fit_x), batch_size=128, shuffle=True)
    model = Autoencoder(len(center))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float('inf')
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, 16):
        model.train()
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = (model(batch) - batch).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation = float(np.mean(score_arrays(model, val, center, scale)))
        training = float(np.mean(losses))
        history.append({'epoch': epoch, 'train_mse': training, 'validation_mse': validation})
        print(f'epoch {epoch}: train={training:.5f}, validation={validation:.5f}', flush=True)
        if validation < best - 1e-5:
            best = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    best_epoch = min(history, key=lambda row: row['validation_mse'])['epoch']
    # Validation cars selected the epoch count; now refit from scratch using all
    # available normal training cars. Test vehicles have not been read yet.
    full = np.concatenate([fit, val])
    center = np.median(full, axis=(0, 1)).astype(np.float32)
    q25, q75 = np.percentile(full, [25, 75], axis=(0, 1))
    scale = np.maximum((q75 - q25).astype(np.float32), 1e-3)
    full_x = torch.from_numpy(((full - center) / scale).transpose(0, 2, 1).copy())
    loader = DataLoader(TensorDataset(full_x), batch_size=128, shuffle=True)
    torch.manual_seed(SEED)
    model = Autoencoder(len(center))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(best_epoch):
        model.train()
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = (model(batch) - batch).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f'refit epoch {epoch + 1}/{best_epoch}: train={np.mean(losses):.5f}', flush=True)
    ROOT.joinpath('models').mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'center': center, 'scale': scale,
                'signals': SIGNALS + ['cell_volt_spread', 'temp_spread'],
                'seed': SEED}, ROOT / 'models' / 'brand2_temporal_autoencoder.pt')
    rows = []
    for path in test_paths:
        arrays, label, split = read_car(path)
        scores = score_arrays(model, arrays, center, scale)
        rows.append({'car': int(path.stem.split('_')[1]), 'label': label,
                     'segments': len(scores), 'score': car_score(scores)})
        print(f'test vehicle {path.stem}: {len(scores)} snippets', flush=True)
    pred = pd.DataFrame(rows)
    result = {'source_split': 'publisher-defined brand2 vehicle train/test',
              'normal_train_vehicles': len(normal), 'fit_vehicles': len(fit_paths),
              'validation_vehicles': len(val_paths), 'test_vehicles': len(test_paths),
              'test_positive_vehicles': int(pred.label.sum()),
              'fit_snippets': len(fit), 'validation_snippets': len(val),
              'refit_snippets': len(full), 'selected_epochs': best_epoch,
              'best_validation_mse': best, 'epochs': history,
              'test_metrics': {'auroc': float(roc_auc_score(pred.label, pred.score)),
                               'average_precision': float(average_precision_score(pred.label, pred.score))}}
    REPORT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pred.to_csv(ROOT / 'reports' / 'temporal_autoencoder_vehicle_scores.csv', index=False)
    print(json.dumps({key: value for key, value in result.items() if key != 'epochs'}, indent=2))


if __name__ == '__main__':
    main()
