"""T02 algorithm: detectors on controlled images, no-sibling fallback, determinism, dev-split regression."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from src import pipeline
from src.qc_features import extract_features, scene_key, score_frame
from tests.test_t02_pipeline import make_ctx


def _base(seed=0, size=(160, 96)) -> Image.Image:
    """A textured scene with real edges (so Laplacian / blockiness are meaningful)."""
    rng = np.random.default_rng(seed)
    a = np.full((size[1], size[0], 3), 120, np.uint8)
    for _ in range(25):
        x, y = rng.integers(0, size[0] - 30), rng.integers(0, size[1] - 20)
        a[y:y + rng.integers(8, 20), x:x + rng.integers(10, 30)] = rng.integers(20, 230, 3)
    return Image.fromarray(a)


def _scene(tmp_path, variants: dict[str, Image.Image], scene="img_00") -> pd.DataFrame:
    rows = []
    for i, (name, im) in enumerate(variants.items()):
        p = tmp_path / f"{scene}_{i}.png"
        im.save(p)
        rows.append({"image_id": f"{scene}_{i}", "path": p.name, "name": name})
    return pd.DataFrame(rows)


def _score(tmp_path, variants, scene="img_00", **kw):
    items = _scene(tmp_path, variants, scene=scene)
    f = score_frame(extract_features(items, tmp_path, **kw))
    return f.merge(items[["image_id", "name"]]).set_index("name")


def test_blur_dark_glare_are_flagged_and_clean_images_are_kept(tmp_path):
    base = _base()
    glare = np.asarray(base).copy()
    glare[10:40, 10:60] = 255                                  # local blown-out patch
    f = _score(tmp_path, {"clean_a": base, "clean_b": base.copy(),
                          "blur": base.filter(ImageFilter.GaussianBlur(4)),
                          "dark": Image.fromarray((np.asarray(base) * 0.15).astype(np.uint8)),
                          "glare": Image.fromarray(glare)})
    assert f.loc[["clean_a", "clean_b"], "pred_action"].tolist() == ["KEEP", "KEEP"]
    assert f.loc["blur", "pred_defect_type"] == "blur"
    assert f.loc["dark", "pred_defect_type"] == "dark"
    assert f.loc["glare", "pred_defect_type"] == "glare"
    assert f.loc[["blur", "dark", "glare"], "score"].min() > f.loc[["clean_a", "clean_b"], "score"].max()


def test_scene_without_siblings_uses_weaker_fallback_and_does_not_crash(tmp_path):
    blurred = _base().filter(ImageFilter.GaussianBlur(4))
    alone = _score(tmp_path, {"x": blurred}, scene="img_01")            # 1 image < min_group -> no sibling reference
    assert alone.loc["x", "has_ref"] == 0 and np.isfinite(alone.loc["x", "score"])
    sib = _score(tmp_path, {"a": _base(), "b": _base(), "c": _base(), "d": blurred}, scene="img_02")
    assert sib["has_ref"].eq(1).all() and sib.loc["d", "pred_defect_type"] == "blur"
    # with siblings the blur evidence is stronger than the down-weighted (x0.5) absolute fallback
    assert sib.loc["d", "s_blur"] > alone.loc["x", "s_blur"]


def test_scene_key_groups_variants_and_tolerates_odd_ids():
    assert scene_key("img_07_3") == "img_07" and scene_key("weird") == "weird"


def test_scoring_is_deterministic(tmp_path):
    items = pd.read_csv("fixtures/t02_raw_qc/images.csv").head(16)
    a = score_frame(extract_features(items, "fixtures/t02_raw_qc")).score.round(9).tolist()
    b = score_frame(extract_features(items, "fixtures/t02_raw_qc")).score.round(9).tolist()
    assert a == b


def test_dev_split_regression_against_own_gold_labels(tmp_path):
    """Frozen thresholds must keep separating dev clean/defect images (labels: reports/my_gold_labels.csv, not gt/)."""
    gold = pd.read_csv("reports/my_gold_labels.csv")
    dev = gold[gold.split == "dev"]
    images = pd.read_csv("fixtures/t02_raw_qc/images.csv")
    ctx = make_ctx(tmp_path)
    scored = pipeline._check_scored(pipeline.process(ctx, {"images": images}), images)
    m = dev.merge(scored[["image_id", "pred_action"]], on="image_id")
    flagged = m.pred_action == "RECAPTURE"
    assert int((flagged & (m.is_defect == 0)).sum()) == 0      # no false alarms on dev
    assert int((~flagged & (m.is_defect == 1)).sum()) == 0     # no misses on dev


def test_poc_pipeline_end_to_end_on_fixtures(tmp_path):
    ctx = make_ctx(tmp_path, params={"budget": 20})
    pipeline.run(ctx)
    pred = pd.read_csv(ctx.output_dir / "ranked_review.csv")
    assert len(pred) == 96 and pred.score.is_monotonic_decreasing
    assert pred.pred_action.isin(["KEEP", "RECAPTURE"]).all()
