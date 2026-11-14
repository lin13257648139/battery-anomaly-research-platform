"""Exploratory condition-aware residual baseline on the fixed brand-2 split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [f"{name}_{stat}" for name in ("soc", "current", "max_temp", "min_temp")
          for stat in ("mean", "std", "delta")]
TARGET = "cell_volt_spread_max"
SEED = 2026


def top_five(values: np.ndarray) -> float:
    size = max(1, int(np.ceil(len(values) * .05)))
    return float(np.partition(values, -size)[-size:].mean())


def model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=80, min_samples_leaf=10,
                               max_features=.8, n_jobs=-1, random_state=SEED)


def main() -> None:
    cols = ["car", "car_label", "source_split", TARGET, *INPUTS]
    frame = pd.read_parquet(ROOT / "features/brand2_snippets.parquet", columns=cols)
    normal = frame[frame.source_split.eq("train") & frame.car_label.eq(0)].copy()
    test = frame[frame.source_split.eq("test")].copy()
    # Each normal vehicle has comparable influence on fitting, regardless of snippet count.
    sampled = pd.concat([g.sample(min(len(g), 250), random_state=SEED)
                         for _, g in normal.groupby("car")], ignore_index=True)
    medians = sampled[INPUTS].median()
    sampled[INPUTS] = sampled[INPUTS].fillna(medians)
    test[INPUTS] = test[INPUTS].fillna(medians)
    if sampled[INPUTS + [TARGET]].isna().any().any() or test[INPUTS + [TARGET]].isna().any().any():
        raise ValueError("Unexpected missing model input")
    cars = np.array(sorted(sampled.car.unique()))
    oof = np.full(len(sampled), np.nan)
    for train_idx, validation_idx in KFold(5, shuffle=True, random_state=SEED).split(cars):
        in_train = sampled.car.isin(cars[train_idx]).to_numpy()
        in_validation = sampled.car.isin(cars[validation_idx]).to_numpy()
        estimator = model()
        estimator.fit(sampled.loc[in_train, INPUTS], sampled.loc[in_train, TARGET])
        oof[in_validation] = estimator.predict(sampled.loc[in_validation, INPUTS])
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete out-of-fold normal predictions")
    normal_residual = np.abs(sampled[TARGET].to_numpy() - oof)
    scale = max(float(np.median(normal_residual)), 1e-6)
    sampled["residual_score"] = normal_residual / scale
    calibration = sampled.groupby("car").residual_score.apply(lambda x: top_five(x.to_numpy()))
    estimator = model()
    estimator.fit(sampled[INPUTS], sampled[TARGET])
    test["residual_score"] = np.abs(test[TARGET].to_numpy() - estimator.predict(test[INPUTS])) / scale
    predictions = test.groupby("car").agg(label=("car_label", "first"),
                                           snippets=(TARGET, "size"))
    predictions["conditioned_score"] = test.groupby("car").residual_score.apply(
        lambda x: top_five(x.to_numpy()))
    predictions["physical_score"] = test.groupby("car")[TARGET].apply(
        lambda x: top_five(x.to_numpy()))
    predictions["train_normal_percentile"] = predictions.conditioned_score.map(
        lambda x: np.searchsorted(np.sort(calibration.to_numpy()), x, side="right") / len(calibration))
    predictions.reset_index().to_csv(ROOT / "reports/conditioned_brand2_vehicle_scores.csv", index=False)
    label = predictions.label.to_numpy()
    result = {
        "protocol": "exploratory, specified after inspecting earlier fixed-test results; independent validation required",
        "training": "normal training vehicles only; 250 snippets maximum per vehicle",
        "normal_training_vehicles": len(cars), "test_vehicles": len(predictions),
        "inputs": INPUTS, "target": TARGET,
        "normal_residual_scale": scale,
        "conditioned": {"auroc": float(roc_auc_score(label, predictions.conditioned_score)),
                        "average_precision": float(average_precision_score(label, predictions.conditioned_score))},
        "physical_same_test": {"auroc": float(roc_auc_score(label, predictions.physical_score)),
                               "average_precision": float(average_precision_score(label, predictions.physical_score))},
        "limitation": "Test labels only used for final evaluation; this is not a validated improvement or online alert rule."
    }
    (ROOT / "reports/conditioned_brand2_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
