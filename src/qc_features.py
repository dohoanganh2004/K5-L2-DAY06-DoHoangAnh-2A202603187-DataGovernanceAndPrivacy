"""T02 feature extraction: blur / dark / glare / compression evidence per image.

Two families of evidence (no ground truth is ever read here):

* absolute  - computed from one image alone (Laplacian variance, brightness,
              saturated fraction, JPEG 8x8 blockiness). Weak across cameras/light.
* relative  - computed against the *scene reference*: the per-pixel median of the
              images that share the same scene key (``img_<scene>_<variant>``).
              Siblings are pixel-aligned, so the reference is a "what this scene
              should look like" image. This is a dataset-structure assumption; when
              a scene has fewer than ``min_group`` images we fall back to absolute
              evidence only (``has_ref = 0``), which is weaker (see reports/analysis_t02.md).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common.image_ops import laplacian_variance, load_rgb, to_gray

_SCENE_RE = re.compile(r"^(?P<scene>.+)_(?P<variant>\d+)$")
GRID = (6, 8)  # rows x cols for local (block) statistics


def scene_key(image_id: str) -> str:
    m = _SCENE_RE.match(str(image_id))
    return m.group("scene") if m else str(image_id)


def blockiness(g: np.ndarray, size: int = 8) -> float:
    """Mean gradient on 8px block borders / elsewhere (JPEG blocking pushes this above 1)."""
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    cx = (np.arange(dx.shape[1]) + 1) % size == 0
    cy = (np.arange(dy.shape[0]) + 1) % size == 0
    border = np.concatenate([dx[:, cx].ravel(), dy[cy, :].ravel()])
    inner = np.concatenate([dx[:, ~cx].ravel(), dy[~cy, :].ravel()])
    return float(border.mean() / (inner.mean() + 1e-6))


def block_means(g: np.ndarray, grid: tuple[int, int] = GRID) -> np.ndarray:
    h, w = g.shape
    rows, cols = grid
    g = g[: h // rows * rows, : w // cols * cols]
    return g.reshape(rows, g.shape[0] // rows, cols, g.shape[1] // cols).mean(axis=(1, 3))


def absolute_features(img: np.ndarray) -> dict:
    g = to_gray(img)
    return {
        "lap": laplacian_variance(img),
        "mean": float(g.mean()),
        "frac_sat": float((g > 245).mean()),
        "block": blockiness(g),
    }


def extract_features(items: pd.DataFrame, image_root, min_group: int = 3) -> pd.DataFrame:
    """items needs image_id + path. Returns one row per image with absolute + relative evidence."""
    items = items.reset_index(drop=True)
    grays, absf = {}, {}
    for r in items.itertuples():
        img = load_rgb(f"{image_root}/{r.path}")
        grays[r.image_id] = to_gray(img)
        absf[r.image_id] = absolute_features(img)
    keys = items.image_id.map(scene_key)
    rows = []
    for key, sub in items.groupby(keys, sort=False):
        ids = list(sub.image_id)
        has_ref = len(ids) >= min_group
        ref_lap = np.median([absf[i]["lap"] for i in ids]) if has_ref else np.nan
        ref_mean = np.median([absf[i]["mean"] for i in ids]) if has_ref else np.nan
        ref_sat = float(np.median([absf[i]["frac_sat"] for i in ids])) if has_ref else 0.0
        ref_block = float(np.median([absf[i]["block"] for i in ids])) if has_ref else 1.0
        for i in ids:
            a = absf[i]
            f = dict(a)
            f["image_id"], f["scene"], f["has_ref"] = i, key, int(has_ref)
            if has_ref:
                f["lap_rel"] = float(np.log((a["lap"] + 1.0) / (ref_lap + 1.0)))        # <0 blur
                f["bright_rel"] = float(np.log((a["mean"] + 1.0) / (ref_mean + 1.0)))   # <0 dark, >0 glare/bloom
                f["sat_rel"] = float(a["frac_sat"] - ref_sat)                           # >0 more clipped highlights
                f["block_rel"] = float(a["block"] / (ref_block + 1e-6))                 # >1 more JPEG blocking
            else:
                f["lap_rel"] = f["bright_rel"] = f["sat_rel"] = 0.0
                f["block_rel"] = 1.0
            rows.append(f)
    return pd.DataFrame(rows).set_index("image_id").loc[items.image_id].reset_index()


# Defaults == the values frozen on the dev split (see tools/calibrate_t02.py and configs/lab.yaml params).
DEFAULT_THRESHOLDS = {
    "blur_lap_rel": -2.5,      # log(lap / scene median) below this = blur
    "dark_bright_rel": -1.0,   # log(mean / scene median) below this = dark
    "glare_sat_rel": 0.005,    # extra fraction of pixels > 245 vs siblings
    "glare_bright_rel": 0.2,   # log brightness gain (global bloom)
    "comp_block_rel": 1.35,    # blockiness ratio vs siblings
    "noref_weight": 0.5,       # confidence multiplier for scenes without siblings (absolute fallback)
    "noref_lap": 1065.5,       # dev clean-image medians (tools/calibrate_t02.py), used only by the no-sibling fallback
    "noref_mean": 117.7,
}
DEFECT_TYPES = ["blur", "dark", "glare", "compression"]


def score_frame(feat: pd.DataFrame, thr: dict | None = None, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Add s_<type> (= evidence / threshold, 1.0 = at threshold), score (max), pred_defect_type, pred_action, reason.

    `drop` removes detectors (ablation only)."""
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    f = feat.copy()
    lap_rel, bright_rel = f.lap_rel.copy(), f.bright_rel.copy()
    noref = f.has_ref == 0
    if noref.any():  # absolute fallback: compare with fixed typical values instead of siblings
        lap_rel[noref] = np.log((f.lap[noref] + 1.0) / (t["noref_lap"] + 1.0))
        bright_rel[noref] = np.log((f["mean"][noref] + 1.0) / (t["noref_mean"] + 1.0))
    w = np.where(noref, t["noref_weight"], 1.0)
    f["s_blur"] = np.clip(lap_rel / t["blur_lap_rel"], 0, None) * w
    f["s_dark"] = np.clip(bright_rel / t["dark_bright_rel"], 0, None) * w
    f["s_glare"] = np.maximum(np.clip(f.sat_rel / t["glare_sat_rel"], 0, None),
                              np.clip(bright_rel / t["glare_bright_rel"], 0, None)) * w
    f["s_compression"] = np.clip((f.block_rel - 1.0) / (t["comp_block_rel"] - 1.0), 0, None) * w
    use = [f"s_{d}" for d in DEFECT_TYPES if d not in drop]
    s = f[use]
    f["score"] = s.max(axis=1)
    top = s.idxmax(axis=1).str.removeprefix("s_")
    flagged = f.score >= 1.0
    f["pred_defect_type"] = np.where(flagged, top, "none")
    f["pred_action"] = np.where(flagged, "RECAPTURE", "KEEP")
    f["reason"] = np.where(flagged, f.pred_defect_type + np.where(noref, " (no reference)", ""), "no_defect_evidence")
    return f
