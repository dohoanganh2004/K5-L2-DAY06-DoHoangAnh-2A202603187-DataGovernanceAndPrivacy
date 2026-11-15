"""T02 analysis for the report: CI, slices, ablations, failure cases (all numbers come from this script).

    python -m tools.analyze_t02          # needs out/baseline, out/poc, out/random (run src.main first)

Outputs: reports/analysis_t02.json, reports/analysis_t02.md, demo/failure_cases_baseline.png

Ablations are REPORTING ONLY: thresholds stay the values frozen on dev (configs/lab.yaml); nothing is re-tuned here.
Top-B follows the official evaluator: eval split only, sort by score descending (stable), B = 20.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common.io import resolve_config
from src.qc_features import DEFAULT_THRESHOLDS, DEFECT_TYPES, extract_features, score_frame

B = 20


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def topb(scores: pd.DataFrame, gold: pd.DataFrame, b: int = B) -> pd.DataFrame:
    """scores: image_id, score. Returns gold-eval rows with a boolean `in_top`."""
    ev = gold[gold.split == "eval"].merge(scores[["image_id", "score"]], on="image_id")
    ev = ev.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    ev["in_top"] = ev.index < b
    return ev


def summarize(ev: pd.DataFrame) -> dict:
    pos = int(ev.is_defect.sum())
    tp = int((ev.in_top & (ev.is_defect == 1)).sum())
    return {
        "recall_at_b": round(tp / pos, 3), "recall_ci95": wilson(tp, pos), "tp_in_top_b": tp, "n_defects": pos,
        "precision_at_b": round(tp / B, 3), "precision_ci95": wilson(tp, B),
        "false_alarms_in_top_b": int((ev.in_top & (ev.is_defect == 0)).sum()),
    }


def slices(ev: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for k, sub in ev.groupby(col):
        pos, neg = sub[sub.is_defect == 1], sub[sub.is_defect == 0]
        rows.append({"slice": f"{col}={k}", "n": int(len(sub)), "n_defects": int(len(pos)),
                     "recall_in_top_b": round(float(pos.in_top.mean()), 3) if len(pos) else None,
                     "false_alarm_in_top_b": round(float(neg.in_top.mean()), 3) if len(neg) else None,
                     "n_false_alarms": int(neg.in_top.sum()), "n_clean": int(len(neg))})
    return rows


def flag_confusion(ev: pd.DataFrame, flagged: pd.Series) -> dict:
    f = ev.image_id.map(flagged).astype(bool)
    return {"tp": int((f & (ev.is_defect == 1)).sum()), "fp": int((f & (ev.is_defect == 0)).sum()),
            "fn": int((~f & (ev.is_defect == 1)).sum()), "tn": int((~f & (ev.is_defect == 0)).sum())}


def montage(rows: pd.DataFrame, root: Path, out: Path, cols: int = 4) -> None:
    W, H = 320, 180
    n = len(rows)
    sheet = Image.new("RGB", (W * cols, (H + 34) * ((n + cols - 1) // cols)), (255, 255, 255))
    for i, r in enumerate(rows.itertuples()):
        im = Image.open(root / r.path).convert("RGB")
        x, y = (i % cols) * W, (i // cols) * (H + 34)
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.text((x + 4, y + H + 3), f"{r.image_id} {r.camera_id}/{r.light}", fill=(0, 0, 0))
        d.text((x + 4, y + H + 18), f"gold={r.defect_type} lap={r.lap:.0f}", fill=(160, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="fixtures/t02_raw_qc")
    ap.add_argument("--gold", default="reports/my_gold_labels.csv")
    ap.add_argument("--config", default="configs/lab.yaml")
    ap.add_argument("--out-dir", default="reports")
    a = ap.parse_args(argv)

    root = Path(a.input)
    gold = pd.read_csv(a.gold)
    images = pd.read_csv(root / "images.csv")
    cfg = resolve_config(a.config, "t02")
    thr = {k: (cfg.get("params") or {}).get(k, v) for k, v in DEFAULT_THRESHOLDS.items()}
    res: dict = {"split": "eval", "n": int((gold.split == "eval").sum()), "budget": B,
                 "n_defects": int(gold[gold.split == "eval"].is_defect.sum()), "thresholds": thr}

    # ---- the three submitted runs, from their own prediction files
    runs = {}
    for name in ["random", "baseline", "poc"]:
        p = pd.read_csv(f"out/{name}/ranked_review.csv")
        runs[name] = topb(p, gold)
        res[name] = summarize(runs[name])
    bset, pset = set(runs["baseline"][runs["baseline"].in_top & (runs["baseline"].is_defect == 1)].image_id), \
        set(runs["poc"][runs["poc"].in_top & (runs["poc"].is_defect == 1)].image_id)
    res["paired_poc_vs_baseline"] = {"only_poc_found": sorted(pset - bset), "only_baseline_found": sorted(bset - pset)}

    # ---- slices (camera is not in the official evaluator, so it is computed here)
    for name in ["baseline", "poc"]:
        res[f"{name}_slices"] = slices(runs[name], "camera_id") + slices(runs[name], "light") + \
            slices(runs[name][runs[name].is_defect == 1].assign(defect=lambda d: d.defect_type), "defect")
    # ---- what the machine would flag outside the human budget (score >= 1.0 / lap < cut)
    poc_pred = pd.read_csv("out/poc/ranked_review.csv").set_index("image_id")
    base_pred = pd.read_csv("out/baseline/ranked_review.csv").set_index("image_id")
    res["flag_confusion_eval"] = {"poc": flag_confusion(runs["poc"], poc_pred.pred_action == "RECAPTURE"),
                                  "baseline": flag_confusion(runs["baseline"], base_pred.pred_action == "RECAPTURE")}

    # ---- ablations (reporting only; thresholds NOT re-tuned)
    abl = {}
    full = extract_features(images, root, min_group=3)
    variants = {"full PoC (sibling reference)": (full, ()),
                "no sibling reference (absolute fallback only)": (extract_features(images, root, min_group=99), ())}
    for d in DEFECT_TYPES:
        variants[f"without {d} detector"] = (full, (d,))
    for name, (feat, drop) in variants.items():
        sc = score_frame(feat, thr, drop=drop)
        ev = topb(sc, gold)
        s = summarize(ev)
        abl[name] = {k: s[k] for k in ["recall_at_b", "recall_ci95", "precision_at_b", "false_alarms_in_top_b"]}
        abl[name]["false_alarm_by_camera"] = {x["slice"]: x["false_alarm_in_top_b"] for x in slices(ev, "camera_id")}
        abl[name]["false_alarm_by_light"] = {x["slice"]: x["false_alarm_in_top_b"] for x in slices(ev, "light")}
        if name.startswith("no sibling"):
            noref_ev = ev
    res["ablation"] = abl

    # ---- failure cases (baseline, and the no-reference PoC), with evidence picture
    feat_abs = full.set_index("image_id")
    def enrich(ev):
        return ev.merge(images[["image_id", "path"]], on="image_id").assign(lap=lambda d: d.image_id.map(feat_abs.lap))
    be = enrich(runs["baseline"])
    fa = be[be.in_top & (be.is_defect == 0)]
    miss = be[~be.in_top & (be.is_defect == 1)]
    res["failure_cases"] = {
        "baseline_false_alarms": fa[["image_id", "camera_id", "light", "lap"]].round(1).to_dict("records"),
        "baseline_misses": miss[["image_id", "camera_id", "light", "defect_type", "lap"]].round(1).to_dict("records"),
        "noref_false_alarms": noref_ev[noref_ev.in_top & (noref_ev.is_defect == 0)][["image_id", "camera_id", "light"]].to_dict("records"),
        "noref_misses": noref_ev[~noref_ev.in_top & (noref_ev.is_defect == 1)][["image_id", "camera_id", "light", "defect_type"]].to_dict("records"),
    }
    # camera softness evidence on CLEAN eval images (absolute Laplacian), computed here from the gold labels
    clean = full.merge(gold[["image_id", "split", "camera_id", "is_defect"]].rename(columns={"camera_id": "cam"}), on="image_id")
    clean = clean[clean.is_defect == 0]
    res["clean_lap_by_camera_all_splits"] = {c: {"n": int(len(g)), "median_lap": round(float(g.lap.median()), 1)}
                                             for c, g in clean.groupby("cam")}
    show = pd.concat([fa.assign(kind="false alarm"), miss.assign(kind="miss")]).head(12)
    montage(show, root, Path("demo/failure_cases_baseline.png"))
    res["failure_montage"] = {"file": "demo/failure_cases_baseline.png", "images": show.image_id.tolist()}

    # ---- runtime / machine, from the run_info files written by src.main
    for name in ["baseline", "poc"]:
        info = json.loads(Path(f"out/{name}/run_info.json").read_text())
        res[f"{name}_runtime"] = {"seconds": info["runtime_seconds"], "n_items": info["stats"].get("n_items"),
                                  "seconds_per_100_items": round(info["runtime_seconds"] / info["stats"]["n_items"] * 100, 3),
                                  "machine": info.get("machine"), "git_sha": info.get("git_sha")}

    out = Path(a.out_dir)
    (out / "analysis_t02.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# T02 analysis (generated by tools/analyze_t02.py)", "",
             f"Eval split: N={res['n']}, defects={res['n_defects']}, B={B}. Gold = `{a.gold}`. Wilson 95% CI.", "",
             "| Run | Recall@20 | 95% CI | Precision@20 | False alarms in top-20 |", "|---|---|---|---|---|"]
    for n in ["random", "baseline", "poc"]:
        r = res[n]
        lines.append(f"| {n} | {r['recall_at_b']} ({r['tp_in_top_b']}/{r['n_defects']}) | {r['recall_ci95']} | {r['precision_at_b']} | {r['false_alarms_in_top_b']} |")
    lines += ["", "## Ablation (thresholds frozen, not re-tuned)", "",
              "| Variant | Recall@20 | 95% CI | False alarms in top-20 | by camera | by light |", "|---|---|---|---|---|---|"]
    for n, r in abl.items():
        lines.append(f"| {n} | {r['recall_at_b']} | {r['recall_ci95']} | {r['false_alarms_in_top_b']} | {r['false_alarm_by_camera']} | {r['false_alarm_by_light']} |")
    (out / "analysis_t02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["random", "baseline", "poc", "paired_poc_vs_baseline", "flag_confusion_eval",
                                           "ablation", "failure_cases", "clean_lap_by_camera_all_splits"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
