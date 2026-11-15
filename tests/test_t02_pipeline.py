"""T02 pipeline plumbing: load validation, process() contract, review override + audit, decision.

`process()` (the algorithm) is replaced by a deterministic stub here, so these tests keep passing
whatever algorithm you implement - they test everything around it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import pipeline
from src.common.context import RunContext
from src.common.review import AuditLog

INPUT = Path("fixtures/t02_raw_qc")


def make_ctx(tmp_path, input_dir=INPUT, **config):
    cfg = {"owner": "tester", "strict_provenance": True, "params": {"budget": 20}} | config
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return RunContext(track="t02", input_dir=Path(input_dir), output_dir=out, config=cfg, seed=42,
                      rng=np.random.default_rng(42), audit=AuditLog(out / "audit_log.jsonl"))


def stub_process(ctx, data):
    """Deterministic fake scores: the first 30 images (by id) are 'defective'."""
    df = data["images"][["image_id"]].copy().sort_values("image_id").reset_index(drop=True)
    df["score"] = np.linspace(1.0, 0.0, len(df))
    df["pred_action"] = np.where(df.index < 30, "RECAPTURE", "KEEP")
    df["reason"] = "stub"
    return df


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(pipeline, "process", stub_process)


# ------------------------------------------------------------------ success paths
def test_end_to_end_writes_all_outputs_and_a_decision(tmp_path, stubbed):
    ctx = make_ctx(tmp_path)
    pipeline.run(ctx)
    for f in ["ranked_review.csv", "review_queue.csv", "contact_sheet.html", "decision.json", "decision_log.csv"]:
        assert (ctx.output_dir / f).exists(), f
    pred = pd.read_csv(ctx.output_dir / "ranked_review.csv")
    assert {"image_id", "score"} <= set(pred.columns) and len(pred) == 96
    assert pred.score.is_monotonic_decreasing                        # ranked best-first
    assert len(pd.read_csv(ctx.output_dir / "review_queue.csv")) == 20   # budget B respected
    d = json.loads((ctx.output_dir / "decision.json").read_text())
    assert d["owner"] == "tester" and d["reason_code"] == "DEFECTS_FLAGGED" and d["canonical"] == "REWORK"
    assert d["metrics"]["n_flagged_outside_review_budget"] == 10     # 30 flagged, 20 reviewable


def test_reviewer_override_is_applied_and_audited(tmp_path, stubbed):
    ctx = make_ctx(tmp_path)
    pipeline.run(ctx)
    q = pd.read_csv(ctx.output_dir / "review_queue.csv", dtype=str).fillna("")
    target = q.loc[0, "item_id"]
    q.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = ["tester", "reject", "KEEP", "looks fine"]
    q.to_csv(ctx.output_dir / "review_queue_reviewed.csv", index=False)
    ctx2 = make_ctx(tmp_path)           # fresh run, same output folder -> picks up the reviewed file
    pipeline.run(ctx2)
    row = pd.read_csv(ctx2.output_dir / "ranked_review.csv").set_index("image_id").loc[target]
    assert row.pred_action == "KEEP" and row.pred_action_machine == "RECAPTURE"   # override applied, machine value kept
    events = [e for e in AuditLog(ctx2.output_dir / "audit_log.jsonl").read() if e["action"] == "override:reject"]
    assert len(events) == 1 and events[0]["actor"] == "tester" and events[0]["item_id"] == target


def test_all_keep_gives_keep_decision(tmp_path, monkeypatch):
    def all_keep(ctx, data):
        df = stub_process(ctx, data)
        df["pred_action"] = "KEEP"
        return df
    monkeypatch.setattr(pipeline, "process", all_keep)
    ctx = make_ctx(tmp_path)
    pipeline.run(ctx)
    d = json.loads((ctx.output_dir / "decision.json").read_text())
    assert d["canonical"] == "GO" and d["reason_code"] == "NO_DEFECT_FLAGGED"


# ------------------------------------------------------------------ failure / edge paths
def test_missing_image_file_fails_clearly(tmp_path):
    bad = tmp_path / "in"
    bad.mkdir()
    pd.DataFrame({"image_id": ["a_0"], "path": ["images/nope.jpg"]}).to_csv(bad / "images.csv", index=False)
    ctx = make_ctx(tmp_path, input_dir=bad, strict_provenance=False)
    with pytest.raises(FileNotFoundError, match="missing"):
        pipeline.load(ctx)


def test_tampered_manifest_hash_is_rejected(tmp_path):
    copy = tmp_path / "in"
    copy.mkdir()
    for f in ["images.csv", "manifest.csv"]:
        (copy / f).write_bytes((INPUT / f).read_bytes())
    (copy / "images.csv").write_text((copy / "images.csv").read_text() + "\n")   # change bytes -> hash mismatch
    with pytest.raises(ValueError, match="provenance"):
        pipeline.load(make_ctx(tmp_path, input_dir=copy))


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns=["pred_action"]), "missing required columns"),
    (lambda d: d.assign(pred_action="MAYBE"), "KEEP or RECAPTURE"),
    (lambda d: d.iloc[:-1], "did not score"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicated"),
    (lambda d: d.assign(score=np.nan), "numeric"),
])
def test_process_contract_violations_are_rejected(tmp_path, mutate, message):
    ctx = make_ctx(tmp_path)
    data = pipeline.load(ctx)
    with pytest.raises(ValueError, match=message):
        pipeline._check_scored(mutate(stub_process(ctx, data)), data["images"])


def test_reviewed_row_without_reviewer_is_rejected(tmp_path, stubbed):
    ctx = make_ctx(tmp_path)
    pipeline.run(ctx)
    q = pd.read_csv(ctx.output_dir / "review_queue.csv", dtype=str).fillna("")
    q.loc[0, "reviewer_decision"] = "approve"          # decision but no reviewer name
    q.to_csv(ctx.output_dir / "review_queue_reviewed.csv", index=False)
    with pytest.raises(ValueError, match="reviewer"):
        pipeline.run(make_ctx(tmp_path))


def test_invalid_override_value_is_rejected(tmp_path, stubbed):
    ctx = make_ctx(tmp_path)
    pipeline.run(ctx)
    q = pd.read_csv(ctx.output_dir / "review_queue.csv", dtype=str).fillna("")
    q.loc[0, ["reviewer", "reviewer_decision", "override_value"]] = ["tester", "edit", "DELETE"]
    q.to_csv(ctx.output_dir / "review_queue_reviewed.csv", index=False)
    with pytest.raises(ValueError, match="KEEP or RECAPTURE"):
        pipeline.run(make_ctx(tmp_path))
