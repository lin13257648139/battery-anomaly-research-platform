"""Describe fixed-threshold screening errors; never tune on held-out cars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features/brand2_snippets.parquet"
SCORES = ROOT / "reports/brand2_vehicle_scores.csv"
OPERATING = ROOT / "reports/operating_point.json"


def main() -> None:
    point = json.loads(OPERATING.read_text(encoding="utf-8"))
    threshold = float(point["threshold"])
    scores = pd.read_csv(SCORES).sort_values("car").copy()
    cols = ["car", "source_split", "soc_mean", "current_mean", "max_temp_mean",
            "cell_volt_spread_max"]
    snippets = pd.read_parquet(FEATURES, columns=cols)
    snippets = snippets[snippets.source_split.eq("test")]
    profiles = []
    for car, group in snippets.groupby("car", sort=True):
        count = max(1, int(np.ceil(len(group) * .05)))
        high = group.nlargest(count, "cell_volt_spread_max")
        profiles.append({"car": int(car), "soc_mean": float(group.soc_mean.mean()),
                         "current_mean": float(group.current_mean.mean()),
                         "max_temp_mean": float(group.max_temp_mean.mean()),
                         "high_score_soc_mean": float(high.soc_mean.mean()),
                         "high_score_current_mean": float(high.current_mean.mean()),
                         "high_score_max_temp_mean": float(high.max_temp_mean.mean())})
    scores = scores.merge(pd.DataFrame(profiles), on="car", validate="one_to_one")
    scores["screen_positive"] = scores.voltage_spread.ge(threshold)
    scores["outcome"] = np.select(
        [(scores.label.eq(1) & scores.screen_positive),
         (scores.label.eq(0) & scores.screen_positive),
         (scores.label.eq(1) & ~scores.screen_positive)],
        ["TP", "FP", "FN"], default="TN")
    counts = scores.outcome.value_counts().to_dict()
    expected = {"TP": point["true_positive"], "FP": point["false_positive"],
                "FN": point["false_negative"], "TN": point["true_negative"]}
    if any(counts.get(name, 0) != value for name, value in expected.items()):
        raise ValueError("Failure analysis does not reproduce the saved operating point")
    output = ROOT / "reports/brand2_failure_cases.csv"
    scores.to_csv(output, index=False)
    summary = {"protocol": "post-hoc descriptive analysis of exploratory terminal screening",
               "threshold_source": "normal training vehicles; threshold not retuned here",
               "threshold": threshold, "counts": expected,
               "failure_vehicles": {kind: scores.loc[scores.outcome.eq(kind), "car"].astype(int).tolist()
                                    for kind in ("FP", "FN")},
               "profile_fields": ["soc_mean", "current_mean", "max_temp_mean",
                                  "high_score_soc_mean", "high_score_current_mean",
                                  "high_score_max_temp_mean"],
               "limitation": "Descriptive test-set inspection only; no causal or generalization claim."}
    (ROOT / "reports/brand2_failure_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
