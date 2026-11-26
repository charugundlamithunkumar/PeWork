"""Grouped model comparison, feature selection and decision thresholds."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix, XGBClassifier

from .features import feature_names

TRACKS = [(mode, rating) for mode in ("app", "context") for rating in (True, False)]


def track_name(mode, ratings):
    return f"{mode}_{'with_rating' if ratings else 'without_rating'}"


def xgb(run):
    return XGBClassifier(n_estimators=20 if run.fixture else 250, max_depth=4,
                         learning_rate=0.1, subsample=0.85, colsample_bytree=0.9,
                         tree_method="hist", device=run.device, n_jobs=4,
                         random_state=run.seed, eval_metric="logloss")


def inputs(frame, kind, features):
    if kind == "tfidf":
        return frame.review_text.tolist()
    return frame.loc[:, features].astype("float32")


def fit_model(run, frame, kind, features):
    if kind == "xgb":
        model = xgb(run)
    elif kind == "tfidf":
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                               min_df=2, sublinear_tf=True),
                              LogisticRegression(max_iter=500, random_state=run.seed))
    elif kind == "rating":
        model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                              LogisticRegression(max_iter=500, random_state=run.seed))
    else:
        model = DummyClassifier(strategy="prior")
    return model.fit(inputs(frame, kind, features), frame.label)


def predict(model, frame, kind, features):
    data = inputs(frame, kind, features)
    if kind == "xgb":
        return model.get_booster().predict(DMatrix(data, nthread=4))
    return model.predict_proba(data)[:, 1]


def shap_values(model, frame, features):
    return model.get_booster().predict(DMatrix(inputs(frame, "xgb", features), nthread=4),
                                       pred_contribs=True)


def threshold_for(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    score = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.lexsort((thresholds, precision, score))[-1]])


def metrics(labels, scores, threshold, probability=False):
    n = len(labels)
    pos = int(np.sum(labels))
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    return {
        "n": n, "positive": pos, "prevalence": pos / n,
        "ap": average_precision_score(labels, scores) if 0 < pos < n else np.nan,
        "roc_auc": roc_auc_score(labels, scores) if 0 < pos < n else np.nan,
        "threshold": threshold, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "brier": brier_score_loss(labels, scores) if probability else np.nan,
    }


def baseline_specs():
    yield "prevalence", "prior", ["rating"]
    yield "rating_only", "rating", ["rating"]
    yield "tfidf_text_only", "tfidf", []
    for mode, ratings in TRACKS:
        yield "full_" + track_name(mode, ratings), "xgb", feature_names(mode, ratings)


def fold_blocks(dev):
    for fold in sorted(dev.fold.unique()):
        yield int(fold), dev.loc[dev.fold.ne(fold)], dev.loc[dev.fold.eq(fold)]


def save_oof(run, filename, frames, dev, model_col="model"):
    scored = pd.concat(frames, ignore_index=True)
    if not scored.groupby(model_col).review_id.agg(lambda ids: ids.is_unique and len(ids) == len(dev)).all():
        raise ValueError("Each model must predict each development review exactly once.")
    run.write_table(filename, scored)


def baselines(run):
    table = run.read_table("amazon_features.parquet")
    dev = table.loc[table.partition.eq("development")]
    rows, oof = [], []
    for fold, train, test in fold_blocks(dev):
        _, cutoff_train, cutoff_test = next(fold_blocks(train))
        for name, kind, features in baseline_specs():
            cutoff_model = fit_model(run, cutoff_train, kind, features)
            threshold = threshold_for(cutoff_test.label, predict(cutoff_model, cutoff_test, kind, features))
            start = perf_counter()
            model = fit_model(run, train, kind, features)
            fit_seconds = perf_counter() - start
            scores = predict(model, test, kind, features)
            rows.append({"model": name, "fold": fold, "feature_count": len(features),
                         "train_seconds": fit_seconds, **metrics(test.label, scores, threshold, True)})
            oof.append(test[["review_id", "label"]].assign(fold=fold, model=name, score=scores,
                                                          threshold=threshold))
    save_oof(run, "baseline_oof.parquet", oof, dev)
    run.write_table("baseline_metrics.csv", pd.DataFrame(rows))


def stratified_sample(frame, count, seed):
    if len(frame) <= count:
        return frame
    return pd.concat([part.sample(min(len(part), max(1, round(count * len(part) / len(frame)))),
                                  random_state=seed) for _, part in frame.groupby("label")])


def rank_and_select(run, dev, columns):
    """Rank and choose k from inner validation; callers own an untouched outer fold."""
    rank_rows = []
    blocks = list(fold_blocks(dev))
    for fold, train, val in blocks:
        model = fit_model(run, train, "xgb", columns)
        sample = stratified_sample(val, 6000, run.seed + fold)
        data = inputs(sample, "xgb", columns)
        def scorer(estimator, x, y):
            return average_precision_score(y, estimator.get_booster().predict(DMatrix(x, nthread=4)))
        importance = permutation_importance(model, data, sample.label, scoring=scorer,
                                            n_repeats=2 if run.fixture else 4, random_state=run.seed)
        shap_sample = sample.iloc[: min(len(sample), 1000)]
        shap = shap_values(model, shap_sample, columns)[:, :-1]
        for feature, drop, impact, direction in zip(columns, importance.importances_mean,
                                                    np.abs(shap).mean(axis=0), shap.mean(axis=0)):
            rank_rows.append({"inner_fold": fold, "feature": feature, "ap_drop": float(drop),
                              "mean_abs_shap": float(impact), "mean_signed_shap": float(direction)})
    ranks = pd.DataFrame(rank_rows)
    order = (ranks.groupby("feature", as_index=False).ap_drop.mean()
             .sort_values(["ap_drop", "feature"], ascending=[False, True]).feature.tolist())
    counts = sorted(set(k for k in (2, 4, 6, 10, 15, 20, len(order)) if k <= len(order)))
    curve = []
    for k in counts:
        scores = []
        elapsed = 0
        for _, train, val in blocks:
            start = perf_counter()
            model = fit_model(run, train, "xgb", order[:k])
            elapsed += perf_counter() - start
            scores.append(average_precision_score(val.label, predict(model, val, "xgb", order[:k])))
        curve.append({"k": k, "inner_mean_ap": float(np.mean(scores)),
                      "inner_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
                      "fit_seconds": elapsed})
    best = max(curve, key=lambda r: r["inner_mean_ap"])
    chosen = next(r["k"] for r in curve if r["inner_mean_ap"] >= best["inner_mean_ap"] - best["inner_se"])
    return order[:chosen], ranks, pd.DataFrame(curve)


def sensitivity(run):
    amz = run.read_table("amazon_features.parquet")
    dev = amz.loc[amz.partition.eq("development")]
    full = run.read_table("baseline_metrics.csv").set_index(["model", "fold"])
    rows, ranks, curves, oof, final = [], [], [], [], {}
    for mode, ratings in TRACKS:
        track = track_name(mode, ratings)
        print(f"Selecting features for {track}", flush=True)
        columns = feature_names(mode, ratings)
        for fold, train, val in fold_blocks(dev):
            selected, ranking, curve = rank_and_select(run, train, columns)
            start = perf_counter()
            model = fit_model(run, train, "xgb", selected)
            seconds = perf_counter() - start
            scores = predict(model, val, "xgb", selected)
            ap = average_precision_score(val.label, scores)
            rows.append({"track": track, "outer_fold": fold, "k": len(selected),
                         "features": json.dumps(selected), "outer_ap": ap,
                         "outer_roc_auc": roc_auc_score(val.label, scores),
                         "ap_delta_vs_full": ap - full.loc[("full_" + track, fold), "ap"],
                         "outer_train_seconds": seconds})
            ranks.append(ranking.assign(track=track, outer_fold=fold))
            curves.append(curve.assign(track=track, outer_fold=fold, selected_k=len(selected)))
            oof.append(val[["review_id", "label"]].assign(model=track, fold=fold, score=scores))
        chosen, ranking, curve = rank_and_select(run, dev, columns)
        final[track] = chosen
        ranks.append(ranking.assign(track=track, outer_fold="final_development"))
        curves.append(curve.assign(track=track, outer_fold="final_development", selected_k=len(chosen)))
    save_oof(run, "selected_oof.parquet", oof, dev)
    run.write_table("selection_by_fold.csv", pd.DataFrame(rows))
    run.write_table("feature_ranks_shap.csv", pd.concat(ranks, ignore_index=True))
    run.write_table("top_k.csv", pd.concat(curves, ignore_index=True))
    run.write_json("selected_features.json", final)
