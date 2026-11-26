"""Reproduce the Week 1 Amazon GPU baseline and the shared reviewer folds."""

from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from .features import DEFERRED, DEFERRED_RATING, REGISTRY
from .io import DEFAULT_DATASET_SUBDIR, Run, digest
from .models import fit_model, metrics, predict, threshold_for
from .person_a_inputs import shared_project


def reviewer_folds(frame):
    folds = np.full(len(frame), -1, dtype=np.int8)
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Sorted integer codes preserve sklearn's group order and avoid quadratic
    # object-string membership checks on hundreds of thousands of reviewers.
    groups, _ = pd.factorize(frame.user_id, sort=True)
    for fold, (_, indices) in enumerate(cv.split(frame, groups=groups)):
        folds[indices] = fold
    if (folds < 0).any() or frame.assign(fold=folds).groupby("user_id").fold.nunique().max() != 1:
        raise ValueError("Incomplete or overlapping reviewer folds.")
    return folds


def audited_amazon(frame, source, raw):
    manifest = source.read_json("data_manifest.json")
    fingerprint = manifest["fingerprints"]["amazon_data.csv"]
    if digest(raw / "amazon_data.csv") != fingerprint:
        raise ValueError("Amazon source differs from the audited split.")
    ids = pd.read_csv(raw / "amazon_data.csv", usecols=["review_id"], dtype="string")
    ids["audit_id"] = [sha256(f"amazon:{fingerprint}:{i}".encode()).hexdigest()[:24] for i in ids.index]
    audited = pd.read_parquet(source.root / "amazon_reviews.parquet", columns=["review_id", "label", "user_id", "prod_id", "rating"])
    audited = audited.rename(columns={"review_id": "audit_id"})
    mapping = audited.merge(ids, on="audit_id", validate="one_to_one").set_index("review_id")
    indexed = frame.set_index("review_id")
    assert len(mapping) == len(audited) and mapping.index.isin(indexed.index).all()
    for col in ["user_id", "prod_id", "label", "rating"]:
        np.testing.assert_array_equal(indexed.loc[mapping.index, col], mapping[col])
    eligible = frame.review_id.isin(mapping.index)
    holdout = frame.prod_id.isin(manifest["partition_summary"]["holdout_ids"])
    assert int((eligible & holdout).sum()) == manifest["partition_summary"]["product_test"]["reviews"]
    return eligible, holdout


def save_splits_and_features(source, raw, work):
    # Explicit engineering allowlist; administrative columns cannot enter XGBoost.
    allowed = set(REGISTRY) | DEFERRED | {"promo_phrase_count"}
    features = [c for c in pq.read_schema(raw / "amazon_features.parquet").names if c in allowed]
    frames = {}
    for platform in ["amazon", "yelp"]:
        ids = ["user_id", "prod_id", "label", "date"]
        if platform == "amazon":
            ids.append("review_id")
        frame = pd.read_parquet(raw / f"{platform}_features.parquet", columns=ids + features)
        assert set(frame.label.unique()) == {-1, 1}
        frame["label"] = frame.label.eq(-1).astype("int8")
        frame.insert(0, "source_row", np.arange(len(frame)))
        if platform == "yelp":
            frame["review_id"] = "yelp-" + frame.source_row.astype(str)
        assert frame.review_id.is_unique and frame[["user_id", "prod_id"]].notna().all().all()
        split = frame[["review_id", "source_row", "user_id", "prod_id"]].copy()
        split["fold"] = reviewer_folds(frame)
        if platform == "amazon":
            split["eligible"], split["product_holdout"] = audited_amazon(frame, source, raw)
            split[["review_id", "prod_id", "eligible", "product_holdout"]].to_parquet(
                work / "data/splits/amazon_product_holdout.parquet", index=False)
        split.to_parquet(work / f"data/splits/{platform}_user_folds.parquet", index=False)
        frame.to_parquet(work / f"data/features/{platform}_person_a.parquet", index=False)
        if platform == "amazon":
            frames["development"] = frame.loc[split.eligible & ~split.product_holdout].assign(fold=split.fold)
            frames["product_holdout"] = frame.loc[split.eligible & split.product_holdout]
        print(f"{platform}: exported {len(frame):,} rows and verified five reviewer-disjoint folds.", flush=True)
    return features, frames


def evaluate(y, scores, cutoff):
    result = metrics(y, scores, cutoff)
    precision, recall, _ = precision_recall_curve(y, scores)
    return {k: v for k, v in result.items() if k != "brier"} | {"pr_auc": auc(recall, precision)}


def train(project=None):
    project = Path(project or Path(__file__).resolve().parents[1])
    shared = shared_project(project)
    source = Run(shared, "amazon_yelp_v2")
    run = Run(project, "person_a_week1", device="cuda")
    raw = shared / DEFAULT_DATASET_SUBDIR
    with TemporaryDirectory(dir=project, prefix="person_a_build_") as temp:
        work = Path(temp)
        for folder in ["data/splits", "data/features", "models/person_a", "results/person_a"]:
            (work / folder).mkdir(parents=True)
        features, frames = save_splits_and_features(source, raw, work)
        dev, hold = frames["development"], frames["product_holdout"]
        assert set(dev.prod_id).isdisjoint(hold.prod_id)
        log, predictions = [], []

        def fit(train_rows):
            model = fit_model(run, train_rows, "xgb", features)
            device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
            if not device.startswith("cuda"):
                raise RuntimeError("XGBoost did not train on CUDA.")
            return model

        def record(name, fold, labels, scores, cutoff):
            log.append({"test_set": name, "fold": fold, **evaluate(labels, scores, cutoff)})

        for outer in range(5):
            validation = (outer + 1) % 5
            training = dev.loc[~dev.fold.isin([outer, validation])]
            val, test = dev.loc[dev.fold.eq(validation)], dev.loc[dev.fold.eq(outer)]
            model = fit(training)
            cutoff = threshold_for(val.label, predict(model, val, "xgb", features))
            scores = predict(model, test, "xgb", features)
            record("amazon_user_cv", outer, test.label, scores, cutoff)
            predictions.append(test[["review_id", "label", "fold"]].assign(score=scores, threshold=cutoff))
            print(f"Finished outer fold {outer}, validation fold {validation}.", flush=True)
        oof = pd.concat(predictions, ignore_index=True)
        assert oof.review_id.is_unique and len(oof) == len(dev)
        pooled = evaluate(oof.label, oof.score, oof.threshold)
        pooled["threshold"] = np.nan
        log.append({"test_set": "amazon_user_cv_pooled", "fold": "all", **pooled})
        oof.to_parquet(work / "results/person_a/oof_predictions.parquet", index=False)

        # The published model has its own fixed validation fold and threshold.
        train_rows, val = dev.loc[dev.fold.ne(0)], dev.loc[dev.fold.eq(0)]
        model = fit(train_rows)
        cutoff = threshold_for(val.label, predict(model, val, "xgb", features))
        path = work / "models/person_a/amazon_xgb.ubj"
        model.save_model(path)
        loaded = XGBClassifier(n_jobs=4)
        loaded.load_model(path)
        sample = val.head(64)
        np.testing.assert_allclose(predict(loaded, sample, "xgb", features),
                                   predict(model, sample, "xgb", features), atol=1e-6)
        scores = predict(loaded, hold, "xgb", features)
        record("amazon_product_holdout", "frozen", hold.label, scores, cutoff)
        hold[["review_id", "label"]].assign(score=scores, threshold=cutoff).to_parquet(
            work / "results/person_a/product_predictions.parquet", index=False)
        log = pd.DataFrame(log).assign(date=pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat(),
            owner="A", experiment_id="W1-A-01", train_set="Amazon eligible, product holdout excluded",
            label_definition="audited Amazon combined label; fake=1",
            feature_set=f"{len(features)} shared notebook predictors", rating_family_included="Y",
            split_file="data/splits/amazon_user_folds.parquet", seed=42,
            notes="Precomputed notebook aggregates use full-platform history. Pooled CV uses fold-specific cutoffs.")
        log.to_csv(work / "results/person_a/results_log.csv", index=False)
        rating_family = {f for f, (_, rated, _) in REGISTRY.items() if rated} | DEFERRED_RATING
        gain = pd.DataFrame({"feature": features, "rating_derived": [f in rating_family for f in features]})
        for kind in ["gain", "total_gain"]:
            gain[kind] = gain.feature.map(model.get_booster().get_score(importance_type=kind)).fillna(0)
            gain[kind + "_share"] = gain[kind] / gain[kind].sum()
        gain["training_distinct_nonmissing"] = [train_rows[c].nunique() for c in features]
        gain.sort_values("total_gain", ascending=False).to_csv(work / "results/person_a/gain_importance.csv", index=False)
        spec = {"features": features, "positive_class": 1, "label": "audited Amazon combined",
                "threshold": cutoff, "weights": path.name, "weights_sha256": digest(path),
                "seed": 42, "device": "cuda", "fit_rows": len(train_rows), "validation_rows": len(val),
                "product_test_rows": len(hold), "validation_fold": 0,
                "weights_input": "raw float32 columns in listed order; preserve missing values",
                "source_parquet_hashes": {p: digest(raw / f"{p}_features.parquet") for p in ["amazon", "yelp"]},
                "rating_family_provisional": sorted(rating_family.intersection(features)),
                "rating_share_normalized_gain": float(gain.loc[gain.rating_derived, "gain_share"].sum()),
                "rating_share_total_gain": float(gain.loc[gain.rating_derived, "total_gain_share"].sum())}
        (path.parent / "model.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        for item in sorted(work.rglob("*")):
            if item.is_file():
                target = project / item.relative_to(work)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Create under the handoff directory so files inherit its access
                # permissions on Windows instead of the private build folder's.
                with NamedTemporaryFile(dir=target.parent, delete=False) as staged:
                    staged_path = Path(staged.name)
                    with item.open("rb") as incoming:
                        copyfileobj(incoming, staged)
                staged_path.replace(target)
        print(log[["test_set", "fold", "n", "roc_auc", "pr_auc", "ap", "precision", "recall", "f1"]].to_string(index=False))
        return log


if __name__ == "__main__":
    train()
