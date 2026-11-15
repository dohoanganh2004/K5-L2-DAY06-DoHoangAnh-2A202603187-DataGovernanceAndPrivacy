"""T02 BASELINE: one global Laplacian-variance threshold, nothing else (as required by docs/tracks/T02.md).

    python -m src.main --track t02 --pipeline src.baseline_laplacian --output out/baseline --run-name baseline --offline --seed 42

score = how blurry the whole image is (higher = blurrier = review first). It cannot see glare, darkness or
compression, and it ignores camera/light - exactly what the PoC has to beat. The cut `lap_threshold`
(configs/lab.yaml) was picked on the dev split only (tools/calibrate_t02.py).
"""
from __future__ import annotations

import pandas as pd

from .common.context import RunContext
from .common.image_ops import laplacian_variance, load_rgb
from . import pipeline


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    images = data["images"]
    cut = float(ctx.param("lap_threshold", 745.0))
    lap = [laplacian_variance(load_rgb(ctx.input_dir / p)) for p in images.path]
    out = pd.DataFrame({"image_id": images.image_id, "lap": lap})
    out["score"] = -out.lap                                   # lower sharpness = more suspicious
    out["pred_action"] = ["RECAPTURE" if v < cut else "KEEP" for v in out.lap]
    out["pred_defect_type"] = ["blur" if v < cut else "none" for v in out.lap]
    out["reason"] = ["low_laplacian" if v < cut else "no_defect_evidence" for v in out.lap]
    return out


def run(ctx: RunContext) -> None:
    """Same load/rank/review/decide plumbing as the PoC; only process() differs."""
    pipeline.run(ctx, process_fn=process)
