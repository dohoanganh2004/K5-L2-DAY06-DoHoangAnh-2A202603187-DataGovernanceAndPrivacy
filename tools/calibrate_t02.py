"""Calibrate / check T02 thresholds on the DEV split only (never touches eval).

    python -m tools.calibrate_t02 [--gold reports/my_gold_labels.csv] [--out reports/calibration_t02.json]

What it does
  1. Computes the scene-relative evidence of every dev image (src/qc_features.py).
  2. For each detector, shows the gap between the weakest true defect of that type and the strongest clean
     image, so the frozen thresholds in configs/lab.yaml can be justified (threshold must sit inside the gap).
  3. Picks the BASELINE threshold (single global Laplacian cut) that maximises F1 on dev.
  4. Prints the clean-image medians used by the no-sibling fallback.
Everything printed here is chosen from dev; eval labels are not even loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.qc_features import DEFAULT_THRESHOLDS, DEFECT_TYPES, extract_features, score_frame


def best_lap_threshold(lap: pd.Series, is_defect: pd.Series) -> tuple[float, float]:
    """Global cut 'lap < t  => defect' with the best F1 on dev. Returns (t, f1)."""
    best = (None, -1.0)
    for t in sorted(set(lap.round(1))):
        pred = lap < t
        tp = int((pred & (is_defect == 1)).sum())
        fp = int((pred & (is_defect == 0)).sum())
        fn = int((~pred & (is_defect == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="fixtures/t02_raw_qc")
    ap.add_argument("--gold", default="reports/my_gold_labels.csv")
    ap.add_argument("--out", default="reports/calibration_t02.json")
    a = ap.parse_args(argv)

    gold = pd.read_csv(a.gold)
    gold = gold[gold.split == "dev"]                       # <- eval is never used here
    items = pd.read_csv(Path(a.input) / "images.csv")
    items = items[items.image_id.isin(gold.image_id)]
    feat = score_frame(extract_features(items, a.input)).merge(gold[["image_id", "defect_type", "is_defect"]], on="image_id")

    report: dict = {"split": "dev", "n": int(len(feat)), "n_defects": int(feat.is_defect.sum()), "detectors": {}}
    clean = feat[feat.is_defect == 0]
    for d in DEFECT_TYPES:
        pos = feat[feat.defect_type == d]
        s = f"s_{d}"
        report["detectors"][d] = {
            "n_defects_of_type": int(len(pos)),
            "weakest_true_defect_score": round(float(pos[s].min()), 3),
            "strongest_clean_score": round(float(clean[s].max()), 3),
            "gap_ok": bool(pos[s].min() > 1.0 > clean[s].max()),      # 1.0 == the frozen threshold
        }
    flagged = feat.pred_action == "RECAPTURE"
    report["dev_confusion_at_frozen_thresholds"] = {
        "tp": int((flagged & (feat.is_defect == 1)).sum()), "fp": int((flagged & (feat.is_defect == 0)).sum()),
        "fn": int((~flagged & (feat.is_defect == 1)).sum()), "tn": int((~flagged & (feat.is_defect == 0)).sum()),
    }
    t, f1 = best_lap_threshold(feat.lap, feat.is_defect)
    report["baseline_lap_threshold"] = {"value": t, "dev_f1": round(f1, 3), "rule": "lap < value => RECAPTURE"}
    report["noref_fallback_values"] = {"noref_lap_clean_median": round(float(clean.lap.median()), 1),
                                       "noref_mean_clean_median": round(float(clean["mean"].median()), 1)}
    report["frozen_thresholds_in_use"] = DEFAULT_THRESHOLDS

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(v["gap_ok"] for v in report["detectors"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
