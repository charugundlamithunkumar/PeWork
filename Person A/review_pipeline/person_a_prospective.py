"""Compare Amazon models with point-in-time inputs and product-grouped selection."""

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from .features import REGISTRY, daily_history, feature_names
from .io import DEFAULT_DATASET_SUBDIR, Run, digest
from .models import predict, threshold_for, xgb
from .person_a import evaluate
from .person_a_inputs import shared_project


LOCAL = feature_names("app", True)
HISTORY = ["prior_count", "prior_mean_stars", "prior_high_share", "prior_low_share",
           "days_since_first", "days_since_last", "abs_prior_star_gap"]
METHODS = {"rating": ("rating",), "local_tree": ("local_tree",),
           "history_tree": ("history_tree",), "word_model": ("word_model",),
           "local_blend": ("local_tree", "word_model"), "history_blend": ("history_tree", "word_model")}


def prepare_reviews(frame):
    """One platform at a time. Include available history plus rows to be scored."""
    frame = frame.reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    if frame[["prod_id", "date", "rating"]].isna().any().any() or not frame.rating.isin(range(1, 6)).all():
        raise ValueError("Each review needs a product ID, date and a rating from 1 to 5.")
    hist = daily_history(frame, "prod_id")
    count = hist.prior_n.replace(0, np.nan)
    frame["prior_count"] = hist.prior_n
    frame["prior_mean_stars"] = hist.prior_stars / count
    frame["prior_high_share"] = hist.prior_pos / count
    frame["prior_low_share"] = hist.prior_neg / count
    frame["days_since_first"] = (frame.date - hist.first_date).dt.days
    frame["days_since_last"] = (frame.date - hist.last_date).dt.days
    frame["abs_prior_star_gap"] = (frame.rating - frame.prior_mean_stars).abs()
    return frame


def words_and_stars(frame):
    # Stateless hashing allows a single cached matrix without learning a
    # vocabulary or document frequencies from validation or held-out reviews.
    vectorizer = HashingVectorizer(n_features=2**18, ngram_range=(1, 2), dtype=np.float32)
    text = vectorizer.transform(frame.review_text.fillna(""))
    stars = frame.rating.to_numpy(dtype=int) - 1
    one_hot = csr_matrix((np.ones(len(frame), dtype=np.float32), (np.arange(len(frame)), stars)),
                         shape=(len(frame), 5))
    return hstack([text, one_hot], format="csr")


def fit_parts(names, frame, matrix, indices, run):
    train = frame.iloc[indices]
    fitted = {}
    for name in names:
        if name == "rating":
            counts = train.groupby("rating").label.agg(["sum", "count"]).reindex(range(1, 6), fill_value=0)
            fitted[name] = (counts["sum"] + 1) / (counts["count"] + 2)
        elif name == "word_model":
            fitted[name] = LogisticRegression(C=1, solver="liblinear", max_iter=200, random_state=42)
            fitted[name].fit(matrix[indices], train.label)
        else:
            columns = LOCAL + (HISTORY if name == "history_tree" else [])
            model = xgb(run)
            model.fit(train[columns].astype("float32"), train.label)
            if not json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("Expected CUDA for XGBoost fitting.")
            fitted[name] = model
    return fitted


def component_scores(fitted, frame, matrix, indices):
    rows = frame.iloc[indices]
    scores = {}
    for name, model in fitted.items():
        if name == "rating":
            scores[name] = rows.rating.map(model).to_numpy()
        elif name == "word_model":
            scores[name] = model.predict_proba(matrix[indices])[:, 1]
        else:
            columns = LOCAL + (HISTORY if name == "history_tree" else [])
            scores[name] = predict(model, rows, "xgb", columns)
    return scores


def combine(method, scores):
    return np.mean([scores[name] for name in METHODS[method]], axis=0)


def score_reviews(directory, source_frame):
    """Score original notebook review rows including strictly prior product history."""
    directory = Path(directory)
    spec = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    path = directory / "model.joblib"
    if digest(path) != spec["sha256"]:
        raise ValueError("Model checksum mismatch.")
    model = joblib.load(path)
    frame = prepare_reviews(source_frame)
    matrix = words_and_stars(frame) if "word_model" in model else None
    scores = combine(spec["method"], component_scores(model, frame, matrix, np.arange(len(frame))))
    return pd.DataFrame({"score": scores, "flagged": scores >= spec["threshold"]}, index=source_frame.index)


def paired_intervals(predictions, rounds=400):
    # Resample whole products. Pre-sort scores once for weighted stepwise AP.
    y = predictions.label.to_numpy()
    group, products = pd.factorize(predictions.prod_id, sort=True)
    schemes = []
    for side in ["week1", "selected"]:
        score = predictions[side].to_numpy()
        order = np.argsort(-score, kind="stable")
        ends = np.r_[np.where(np.diff(score[order]) != 0)[0], len(y)-1]
        positive = predictions[side + "_flag"].to_numpy(dtype=bool)
        schemes.append((order, ends, positive))
    def measures(w, scheme):
        order, ends, positive = scheme
        tp = np.cumsum((w*y)[order])[ends]
        totals = np.cumsum(w[order])[ends]
        precision = np.divide(tp, totals, out=np.zeros(len(tp)), where=totals>0)
        ap = np.dot(np.diff(np.r_[0, tp]), precision) / tp[-1]
        f1 = 2*np.sum(w*y*positive) / (np.sum(w*y) + np.sum(w*positive))
        return np.array([ap, f1])
    unit = np.ones(len(y), dtype=int)
    base, selected = [measures(unit, scheme) for scheme in schemes]
    for idx, side in enumerate(["week1", "selected"]):
        np.testing.assert_allclose([base, selected][idx][0], average_precision_score(y, predictions[side]))
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(rounds):
        w = np.bincount(rng.integers(len(products), size=len(products)), minlength=len(products))[group]
        if 0 < np.sum(w*y) < np.sum(w):
            deltas.append(measures(w, schemes[1])-measures(w, schemes[0]))
    lo, hi = np.quantile(deltas, [0.025, 0.975], axis=0)
    return pd.DataFrame({"metric": ["average_precision", "f1"], "delta": selected-base,
                         "lower_95": lo, "upper_95": hi, "product_clusters": len(products), "draws": len(deltas)})


def removal_check(source, directory):
    """Secondary readout only. Keep the combined-label decision cutoffs."""
    observed = pd.read_parquet(source, columns=["review_id", "review_is_removed_by_amazon"])
    saved = pd.read_parquet(directory / "product_holdout_predictions.parquet")
    joined = saved.merge(observed, on="review_id", validate="one_to_one", how="left")
    known = joined.dropna(subset=["review_is_removed_by_amazon"])
    assert set(known.review_is_removed_by_amazon.unique()) == {0, 1}
    spec = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    records = []
    for name, threshold in [("selected", spec["threshold"]), ("week1", saved.threshold.iloc[0])]:
        records.append({"model": name, "endpoint": "observed Amazon removal",
                        **evaluate(known.review_is_removed_by_amazon.astype(int), known[name], threshold)})
    report = pd.DataFrame(records)
    report.to_csv(directory / "removal_check.csv", index=False)
    return report


def export_tree_gain(directory):
    fitted = joblib.load(directory / "model.joblib")
    for name in ("local_tree", "history_tree"):
        if name in fitted:
            booster = fitted[name].get_booster()
            rated = {col for col in LOCAL if REGISTRY[col][1]}
            rated.update(["prior_mean_stars", "prior_high_share", "prior_low_share", "abs_prior_star_gap"])
            table = pd.DataFrame({"feature": booster.feature_names})
            table["rating_derived"] = table.feature.isin(rated)
            for kind in ["gain", "total_gain"]:
                values = table.feature.map(booster.get_score(importance_type=kind)).fillna(0)
                table[kind + "_share"] = values / values.sum()
            table.sort_values("total_gain_share", ascending=False).to_csv(directory / "tree_gain.csv", index=False)


def run_experiment(root):
    root = Path(root)
    run = Run(root, "person_a_prospective", device="cuda")
    source = shared_project(root) / DEFAULT_DATASET_SUBDIR / "amazon_features.parquet"
    old_spec = json.loads((root / "models/person_a/model.json").read_text())
    assert digest(source) == old_spec["source_parquet_hashes"]["amazon"]
    raw = pd.read_parquet(source, columns=["review_id", "user_id", "prod_id", "date", "review_text", "label", *LOCAL])
    split = pd.read_parquet(root / "data/splits/amazon_user_folds.parquet")
    assert raw.review_id.is_unique and set(raw.label.unique()) == {-1, 1}
    raw = raw.merge(split[["review_id", "user_id", "prod_id", "fold", "eligible", "product_holdout"]],
                    on=["review_id", "user_id", "prod_id"], validate="one_to_one", how="left")
    assert raw.fold.notna().all() and len(raw) == len(split)
    raw["label"] = raw.label.eq(-1).astype(int)
    frame = prepare_reviews(raw).loc[raw.eligible].reset_index(drop=True)
    del raw
    # A small out-of-order example verifies both same-day exclusion
    # and invariance to later reviews, irrespective of input ordering.
    probe = pd.DataFrame({"prod_id": ["a"]*5, "rating": [2,5,4,1,5],
                          "date": pd.to_datetime(["2020-02-01","2020-01-01","2020-02-01","2020-04-01","2020-01-01"])})
    hist = prepare_reviews(probe)
    np.testing.assert_array_equal(hist.prior_count, [2,0,2,4,0])
    np.testing.assert_allclose(hist.prior_mean_stars.iloc[[0,2]], [5,5])
    pd.testing.assert_frame_equal(hist.loc[:2,HISTORY], prepare_reviews(probe.drop(index=3)).loc[:2,HISTORY])
    matrix = words_and_stars(frame)
    development = ~frame.product_holdout
    fit = development & frame.fold.ne(0)
    cutoff_date = frame.loc[fit, "date"].quantile(0.8).normalize()
    earlier = frame.date.lt(cutoff_date)
    pool = np.flatnonzero(fit & earlier)
    groups, _ = pd.factorize(frame.prod_id.iloc[pool], sort=True)
    choices = []
    groups_cv = GroupKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (tr, val) in enumerate(groups_cv.split(pool, groups=groups)):
        train_idx, test_idx = pool[tr], pool[val]
        assert set(groups[tr]).isdisjoint(groups[val])
        fitted = fit_parts(["rating", "local_tree", "history_tree", "word_model"], frame, matrix, train_idx, run)
        values = component_scores(fitted, frame, matrix, test_idx)
        for method in METHODS:
            choices.append({"fold": fold, "method": method, "rows": len(test_idx),
                            "ap": average_precision_score(frame.label.iloc[test_idx], combine(method, values))})
        print(f"Completed development product fold {fold+1}/3", flush=True)
    choices = pd.DataFrame(choices)
    winner = choices.groupby("method", sort=False).ap.mean().idxmax()
    directory = root / "results/person_a" / datetime.now(timezone.utc).strftime("prospective_%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    choices.to_csv(directory / "selection.csv", index=False)
    results = []
    print(f"Selected {winner} on earlier development products.", flush=True)
    for test, training, validation, testing in [
        ("forward_time", fit & earlier, development & frame.fold.eq(0) & earlier, development & ~earlier),
        ("product_holdout", fit, development & frame.fold.eq(0), frame.product_holdout)]:
        tr, val, te = map(np.flatnonzero, [training, validation, testing])
        assert set(frame.user_id.iloc[tr]).isdisjoint(frame.user_id.iloc[val])
        if test == "forward_time":
            assert frame.date.iloc[np.r_[tr,val]].max() < frame.date.iloc[te].min()
        else:
            assert set(frame.prod_id.iloc[np.r_[tr,val]]).isdisjoint(frame.prod_id.iloc[te])
        parts = fit_parts(list(dict.fromkeys([*METHODS[winner], "rating"])), frame, matrix, tr, run)
        v, t = [component_scores(parts, frame, matrix, ix) for ix in (val,te)]
        rows = frame.iloc[te][["review_id","prod_id","date","label"]].copy()
        for method in dict.fromkeys(["rating", winner]):
            threshold = threshold_for(frame.label.iloc[val], combine(method, v))
            scores = combine(method, t)
            results.append({"cohort": test, "model": method, **evaluate(rows.label, scores, threshold)})
        rows["selected"], rows["selected_flag"] = scores, scores >= threshold
        if test == "product_holdout":
            old = pd.read_parquet(root / "results/person_a/product_predictions.parquet")
            rows = rows.merge(old.rename(columns={"score":"week1"}), on=["review_id","label"], validate="one_to_one")
            assert len(rows) == len(old) == len(te) and rows.threshold.nunique() == 1
            rows["week1_flag"] = rows.week1 >= rows.threshold
            results.append({"cohort": test, "model":"week1_retrospective",
                            **evaluate(rows.label, rows.week1, rows.threshold.iloc[0])})
            paired_intervals(rows).to_csv(directory / "paired_intervals.csv", index=False)
            joblib.dump({k: parts[k] for k in METHODS[winner]}, directory / "model.joblib", compress=3)
            spec = {"method": winner, "threshold": threshold, "sha256": digest(directory / "model.joblib"),
                    "candidate_weights": "equal within each blend", "selection": "mean fold AP on earlier development products",
                    "time_test_from": str(cutoff_date.date()), "fit_rows": len(tr), "threshold_rows": len(val),
                    "holdout_rows": len(te), "split_sha256": digest(root / "data/splits/amazon_user_folds.parquet"),
                    "source_sha256": old_spec["source_parquet_hashes"]["amazon"],
                    "label": "audited Amazon combined, positive=1", "input_features": LOCAL,
                    "history_features": HISTORY, "history": "all source activity for product on strictly earlier calendar days",
                    "extractor": "original notebook per-review columns plus prepare_reviews; include review_text"}
            (directory / "model.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
            # Check external scoring entry point using original rows for one held-out product.
            prod = rows.prod_id.iloc[0]
            sample = pd.read_parquet(source, columns=["review_id","prod_id","date","review_text",*LOCAL],
                                     filters=[("prod_id","=",prod)])
            scored = score_reviews(directory, sample).assign(review_id=sample.review_id)
            check = rows.loc[rows.prod_id.eq(prod)].merge(scored, on="review_id", validate="one_to_one")
            assert len(check) == rows.prod_id.eq(prod).sum()
            np.testing.assert_allclose(check.score, check.selected, atol=1e-6)
            np.testing.assert_array_equal(check.flagged, check.selected_flag)
        rows.to_parquet(directory / (test + "_predictions.parquet"), index=False)
    report = pd.DataFrame(results)
    report.to_csv(directory / "metrics.csv", index=False)
    removal_check(source, directory)
    export_tree_gain(directory)
    return directory, report
