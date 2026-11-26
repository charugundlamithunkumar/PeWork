"""Run from the project root with python -m "Person C.run_transfer"."""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

history_model = import_module("Person A.review_pipeline.person_a_prospective")
ablation = import_module("Person B.review_pipeline.person_b_ablation")
TOP3 = import_module("Person B.review_pipeline.person_b_trimmed_ablation").TOP3

C = Path(__file__).resolve().parent
ROOT = C.parent
A = ROOT / "Person A"
B = ROOT / "Person B/results/person_b"
SPLITS = A / "data/splits"
HISTORY = A / "results/person_a/prospective_20261121T180956177896Z"
MODELS = C / "models"


def read_spec(path):
    return json.loads(path.read_text(encoding="utf-8"))


def match_hash(path, expected):
    if ablation.sha256(path) != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def indexed(frame):
    if frame.review_id.hasnans or not frame.review_id.is_unique:
        raise ValueError("Expected one nonmissing key per review")
    return frame.set_index("review_id")


def model_specs(spec, dev, train_variants):
    MODELS.mkdir(exist_ok=True)
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    b_path = B / "models/minus_family_32.json"
    without_family = read_spec(b_path)["features"]
    paths = {"full_45_frozen": A / "models/person_a/model.json", "no_family_32": b_path}
    for name, base, model_id in [
        ("full_44_no_prod_review_count", spec["features"], "C-44-CPU"),
        ("no_family_31_no_prod_review_count", without_family, "C-31-CPU"),
    ]:
        path = MODELS / (name + ".json")
        if train_variants:
            features = [f for f in base if f != "prod_review_count"]
            model = ablation.fit(features, train, "cpu")
            threshold = ablation.best_f1_threshold(val.label, ablation.scorer(model, features)(val))
            ablation.freeze(model, features, name, model_id, threshold, A, MODELS, "cpu")
        paths[name] = path
    rating_path = MODELS / "rating_only_table.json"
    if train_variants:
        rates = train.groupby("rating").label.mean()
        cut = ablation.best_f1_threshold(val.label, val.rating.map(rates))
        rating_path.write_text(json.dumps({"rates": rates.to_dict(), "threshold": cut,
            "features": ["rating"], "fit_folds": [1, 2, 3, 4], "validation_fold": 0,
            "split_sha256": ablation.sha256(SPLITS / "amazon_user_folds.parquet")}, indent=2))
    return paths, rating_path


def get_scores(spec, dev, amazon, yelp, dataset, train_variants):
    paths, rating_path = model_specs(spec, dev, train_variants)
    frames = {"amazon": amazon, "yelp": yelp}
    scores = {key: frame[["review_id", "label"]].copy() for key, frame in frames.items()}
    info = {}
    for name, path in paths.items():
        bundle = read_spec(path)
        for split, checksum in bundle.get("split_sha256", {}).items():
            match_hash(SPLITS / split, checksum)
        weights = path.parent / bundle["weights"]
        match_hash(weights, bundle["weights_sha256"])
        booster = Booster()
        booster.load_model(weights)
        if booster.feature_names != bundle["features"]:
            raise ValueError(f"Incorrect input order in {path}")
        for key, frame in frames.items():
            scores[key][name] = booster.predict(DMatrix(frame[bundle["features"]].astype("float32")))
        info[name] = (bundle.get("model_id", "W1-A-01"), bundle["threshold"], weights,
                      "full" if "full_" in name else "removed")

    rating = read_spec(rating_path)
    match_hash(SPLITS / "amazon_user_folds.parquet", rating["split_sha256"])
    rates = {int(k): v for k, v in rating["rates"].items()}
    for key, frame in frames.items():
        scores[key]["rating_only_table"] = frame.rating.map(rates)
    info["rating_only_table"] = ("W1-B-05", rating["threshold"], rating_path, "rating only")
    history = read_spec(HISTORY / "model.json")
    match_hash(SPLITS / "amazon_user_folds.parquet", history["split_sha256"])
    for key, frame in frames.items():
        path = dataset / (key + "_features.parquet")
        match_hash(path, spec["source_parquet_hashes"][key])
        columns = list(dict.fromkeys(["prod_id", "date", "user_id", "label", "review_text",
                                     *history["input_features"]] + (["review_id"] if key == "amazon" else [])))
        raw = pd.read_parquet(path, columns=columns)
        if key == "yelp":
            raw["review_id"] = "yelp-" + pd.Series(np.arange(len(raw)), index=raw.index).astype(str)
        subset = raw.iloc[frame.source_row.to_numpy()]
        for field in ["review_id", "user_id", "prod_id", "date", "rating"]:
            np.testing.assert_array_equal(subset[field].to_numpy(), frame[field].to_numpy())
        np.testing.assert_array_equal(subset.label.eq(-1), frame.label)
        # Score the complete source timeline before restricting to any evaluation population.
        scores[key]["earlier_history"] = history_model.score_reviews(HISTORY, raw).score.to_numpy()[frame.source_row]
    info["earlier_history"] = ("W1-A-03", history["threshold"], HISTORY / "model.joblib", "partial")

    for name, file, col in [
        ("full_45_frozen", A / "results/person_a/product_predictions.parquet", "score"),
        ("earlier_history", HISTORY / "product_holdout_predictions.parquet", "selected"),
        ("no_family_32", B / "minus_family_32_holdout.parquet", "score"),
    ]:
        original = indexed(pd.read_parquet(file))
        observed = indexed(scores["amazon"])
        if set(observed.index) != set(original.index):
            raise ValueError(f"Holdout review IDs differ for {name}")
        original = original.loc[observed.index]
        np.testing.assert_array_equal(original.label, observed.label)
        np.testing.assert_array_equal(original[col], observed[name])
        if name == "no_family_32":
            np.testing.assert_array_equal(original.flagged, observed[name] >= info[name][1])
    return scores, info


def run(dataset, train_variants):
    spec, _, dev, amazon = ablation.load(A)
    yelp = pd.read_parquet(A / "data/features/yelp_person_a.parquet")
    scores, info = get_scores(spec, dev, amazon, yelp, dataset, train_variants)
    unseen = ~amazon.user_id.isin(dev.user_id)
    trim = ~amazon.prod_id.isin(TOP3)
    cohorts = {"amazon_holdout_all": amazon, "amazon_holdout_new_reviewers": amazon[unseen],
               "amazon_holdout_minus3": amazon[trim], "amazon_holdout_minus3_new": amazon[trim & unseen],
               "yelp_all": yelp, "yelp_inflation": yelp[yelp.label.eq(0) | yelp.rating.ge(4)]}
    rows = []
    for cohort, frame in cohorts.items():
        keys = indexed(pd.read_parquet(C / "test_sets" / (cohort + ".parquet")))
        expected = indexed(frame).label.sort_index()
        pd.testing.assert_series_equal(keys.label.sort_index(), expected, check_dtype=False)
        table = scores[cohort.split("_")[0]].set_index("review_id").loc[keys.index]
        for name, (model_id, cut, path, family) in info.items():
            values = table[name].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Missing scores for {name} on {cohort}")
            metrics = ablation.report(keys.label, values, cut)
            rows.append({"test_set": cohort, "model": name, **metrics, "positives": metrics["positive"],
                         "flag_rate": np.mean(values >= cut), "rating_family_included": family != "removed",
                         "rating_family_scope": family, "owner": "C", "date": pd.Timestamp.today().date(),
                         "model_id": model_id, "artifact": path.relative_to(ROOT).as_posix(),
                         "weights_sha256": ablation.sha256(path),
                         "split_file": (SPLITS / "amazon_user_folds.parquet").relative_to(ROOT).as_posix(),
                         "cohort_file": (C / "test_sets" / (cohort + ".parquet")).relative_to(ROOT).as_posix(), "seed": 42,
                         "notes": "Frozen Amazon scoring and validation threshold; no target tuning"})
    result = pd.DataFrame(rows)
    metrics = ["threshold", "ap", "pr_auc", "roc_auc", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    b_metrics = pd.read_csv(B / "ablation_holdout_cohorts.csv").query(
        "model == 'minus_family_32' and cohort == 'all holdout'")
    c_metrics = result.query("model == 'no_family_32' and test_set == 'amazon_holdout_all'")
    np.testing.assert_allclose(c_metrics[metrics], b_metrics[metrics], rtol=0, atol=1e-12)
    for key, table in scores.items():
        table.to_parquet(C / (key + "_predictions.parquet"), index=False)
    result.to_csv(C / "transfer_results.csv", index=False)
    print(result.loc[result.model.isin(["earlier_history", "rating_only_table"]),
                     ["test_set", "model", "ap", "roc_auc", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/ECommerce_Project")
    parser.add_argument("--train-variants", action="store_true", help="Fit C's two count ablations and rating reference on Amazon")
    args = parser.parse_args()
    run(args.dataset, args.train_variants)
