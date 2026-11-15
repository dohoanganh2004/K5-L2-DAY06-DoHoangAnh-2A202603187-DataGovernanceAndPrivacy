"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv) + images.csv validation
    02 process   your algorithm / model (the research part)         <- YOU implement this one
    03 rank      prioritise for a fixed review budget B, write review_queue.csv + contact_sheet.html
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    KEEP / RECAPTURE (= GO / REWORK) with reason_code + owner, write outputs

Required output: out/ranked_review.csv with columns image_id, score (docs/tracks/T02.md).
Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .qc_features import DEFAULT_THRESHOLDS, extract_features, score_frame

ACTIONS = {"KEEP", "RECAPTURE"}
META_COLS = ["path", "camera_id", "light", "split"]     # copied from images.csv onto every scored row


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images_csv = ctx.input_dir / "images.csv"
    if not images_csv.exists():
        raise FileNotFoundError(f"T02 needs {images_csv} (columns image_id, path, camera_id, light, split)")
    images = pd.read_csv(images_csv)
    missing_cols = [c for c in ["image_id", "path"] if c not in images.columns]
    if missing_cols:
        raise ValueError(f"{images_csv} is missing required columns {missing_cols}")
    if images.image_id.duplicated().any():
        raise ValueError(f"{images_csv} has duplicated image_id values")
    absent = [p for p in images.path if not (ctx.input_dir / p).exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} image files listed in images.csv are missing, first: {absent[:3]}")
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None, "images": images}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """PoC algorithm (see src/qc_features.py): per-image blur / dark / glare / compression evidence measured against the
    pixel-aligned sibling images of the same scene, scaled by thresholds frozen on dev; score = strongest evidence.

    Contract. Input: data["images"] (image_id, path, camera_id, light, split); images live under ctx.input_dir.

    Contract for the returned DataFrame (one row per image in data["images"], no duplicates):
        image_id    str    id from images.csv
        score       float  higher = more suspicious (reviewed first; this is what Recall@B ranks on)
        pred_action str    "RECAPTURE" if you call the image defective, else "KEEP"
        reason      str    (recommended) short why, e.g. "blur", shown in the review queue
        pred_defect_type  (optional) one of none/blur/dark/glare/compression, scored as secondary metric
    Other columns are kept. path/camera_id/light/split are added for you after this function returns.
    Use ctx.param("name", default) for thresholds so they live in configs/lab.yaml (freeze them before eval).
    """
    images = data["images"]
    thr = {k: ctx.param(k, v) for k, v in DEFAULT_THRESHOLDS.items()}     # frozen on dev, see configs/lab.yaml
    feats = extract_features(images, ctx.input_dir, min_group=int(ctx.param("min_group", 3)))
    ctx.stats["n_no_reference"] = int((feats.has_ref == 0).sum())          # scenes too small for sibling reference
    scored = score_frame(feats, thr)
    keep = ["image_id", "score", "pred_action", "pred_defect_type", "reason",
            "s_blur", "s_dark", "s_glare", "s_compression", "has_ref"]
    return scored[keep]


def _check_scored(scored: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly (exit 2) if process() broke the contract, then attach image metadata."""
    need = ["image_id", "score", "pred_action"]
    missing = [c for c in need if c not in scored.columns]
    if missing:
        raise ValueError(f"process() output is missing required columns {missing}")
    if scored.image_id.duplicated().any():
        raise ValueError("process() output has duplicated image_id values")
    unknown = sorted(set(scored.image_id.astype(str)) - set(images.image_id.astype(str)))
    if unknown:
        raise ValueError(f"process() returned image_id not present in images.csv, first: {unknown[:3]}")
    absent = sorted(set(images.image_id.astype(str)) - set(scored.image_id.astype(str)))
    if absent:
        raise ValueError(f"process() did not score {len(absent)} images, first: {absent[:3]}")
    if not pd.api.types.is_numeric_dtype(scored.score) or scored.score.isna().any():
        raise ValueError("process() `score` must be numeric with no NaN")
    bad = sorted(set(scored.pred_action.astype(str)) - ACTIONS)
    if bad:
        raise ValueError(f"pred_action must be KEEP or RECAPTURE, got {bad}")
    out = scored.copy()
    out["image_id"] = out.image_id.astype(str)
    meta = [c for c in META_COLS if c in images.columns and c not in out.columns]
    return out.merge(images[["image_id"] + meta].astype({"image_id": str}), on="image_id", how="left")


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B by `score` -> review_queue.csv + offline HTML contact sheet (reviewer picks, then Export CSV)."""
    budget = int(ctx.param("budget", 20))
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=budget,
                          reason_col="reason" if "reason" in scored.columns else None,
                          action_col="pred_action",
                          slice_col="light" if "light" in scored.columns else None,
                          evidence_col="path" if "path" in scored.columns else None)
    write_review_queue(q, ctx.out("review_queue.csv"))
    if "path" in scored.columns:
        render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                             title=f"T02 review queue (top {budget})")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply out/review_queue_reviewed.csv if it exists; otherwise pass through (machine-only).

    approve / skip / escalate : keep the machine action, just log who decided what.
    edit / reject + override_value (KEEP|RECAPTURE) : replace pred_action; raw prediction is never modified in place
    (the original is kept in `pred_action_machine`), and every change is appended to audit_log.jsonl.
    """
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    out = scored.copy()
    out["pred_action_machine"] = out["pred_action"]
    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return out
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    unknown = sorted(set(reviewed.item_id) - set(out.image_id))
    if unknown:
        raise ValueError(f"review file refers to image_id not in the run, first: {unknown[:3]}")
    bad = sorted(set(reviewed.override_value[reviewed.override_value != ""]) - ACTIONS)
    if bad:
        raise ValueError(f"override_value must be KEEP or RECAPTURE, got {bad}")
    for r in reviewed.itertuples(index=False):
        if r.reviewer_decision in {"approve", "skip", "escalate"}:
            ctx.audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id, reason=r.review_note)
    return apply_overrides(out, reviewed, id_col="image_id", value_col="pred_action", audit=ctx.audit)


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Batch decision. Policy (frozen before eval): any image still marked RECAPTURE after review -> RECAPTURE (=REWORK),
    otherwise KEEP (=GO). Flags outside the top-B were never seen by a human; that is stated in the reason."""
    n = len(final)
    flagged = final.pred_action == "RECAPTURE"
    budget = int(ctx.param("budget", 20))
    queue_ids = set(pd.read_csv(ctx.out("review_queue.csv")).item_id.astype(str))
    outside = int((flagged & ~final.image_id.isin(queue_ids)).sum())
    metrics = {"n_images": n, "n_recapture": int(flagged.sum()), "n_flagged_outside_review_budget": outside,
               "budget": budget, "n_reviewed": int(ctx.stats.get("n_reviewed", 0))}
    evidence = [str(ctx.out("review_queue.csv")), str(ctx.out("audit_log.jsonl"))]
    owner = ctx.config.get("owner", "TODO-owner")
    if not flagged.any():
        return Decision("KEEP", "NO_DEFECT_FLAGGED", "No image is marked for recapture", owner, ctx.track, evidence, metrics)
    reason = f"{int(flagged.sum())}/{n} images marked for recapture"
    if outside:
        reason += f"; {outside} of them are outside the top-{budget} human review budget (machine-only decision)"
    if not ctx.stats.get("n_reviewed"):
        reason += "; no human review file was applied"
    return Decision("RECAPTURE", "DEFECTS_FLAGGED", reason, owner, ctx.track, evidence, metrics)


def run(ctx: RunContext, process_fn=None) -> None:
    """`process_fn` lets another module (e.g. the baseline) reuse the same load/rank/review/decide plumbing."""
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = _check_scored((process_fn or process)(ctx, data), data["images"])
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final = final.sort_values("score", ascending=False, kind="mergesort")
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
