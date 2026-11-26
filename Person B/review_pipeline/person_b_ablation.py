"""Paired rating-family ablation on the Person A splits (Workstream B, task 6).

    python -m review_pipeline.person_b_ablation --package "../Person A" [--device cpu|cuda]

Protocol (from Person A's handoff, unchanged):
  * development = eligible & ~product_holdout, five reviewer-disjoint folds, seed 42
  * outer test fold k: threshold chosen on fold (k+1) % 5, fit on the other three folds
  * frozen-style fit: folds 1-4, threshold on fold 0, scored on the product holdout
  * XGBoost settings identical to review_pipeline.models.xgb (250 trees, depth 4, lr 0.1)
Only the feature set changes between rows.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from xgboost import Booster, DMatrix, XGBClassifier

from .rating_family import FAMILY, FAMILY_LIST, NOT_IN_FAMILY_BUT_RELATED, TIER1_LIST, TIER2_LIST

from . import default_package

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "results/person_b"


def best_f1_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.lexsort((thresholds, precision, f1))[-1]])


def report(y, score, threshold=None, flagged=None):
    y, score = np.asarray(y), np.asarray(score)
    flagged = (score >= threshold) if flagged is None else np.asarray(flagged)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, score)
    return {"n": len(y), "positive": int(y.sum()), "prevalence": y.mean(),
            "ap": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            "pr_auc": auc(recall, precision),
            "threshold": threshold if threshold is not None else np.nan,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def load(package):
    package = Path(package)
    spec = json.loads((package / "models/person_a/model.json").read_text())
    feats = pd.read_parquet(package / "data/features/amazon_person_a.parquet")
    folds = pd.read_parquet(package / "data/splits/amazon_user_folds.parquet", columns=["review_id", "fold", "eligible"])
    hold = pd.read_parquet(package / "data/splits/amazon_product_holdout.parquet", columns=["review_id", "product_holdout"])
    frame = (feats.merge(folds, on="review_id", validate="one_to_one")
                  .merge(hold, on="review_id", validate="one_to_one"))
    dev = frame[frame.eligible & ~frame.product_holdout].reset_index(drop=True)
    test = frame[frame.eligible & frame.product_holdout].reset_index(drop=True)
    assert len(dev) == spec["fit_rows"] + spec["validation_rows"] and len(test) == spec["product_test_rows"]
    assert dev.groupby("user_id").fold.nunique().max() == 1
    return spec, frame, dev, test


def feature_sets(features):
    assert set(FAMILY_LIST) <= set(features), "family names must all be model features"
    keep = lambda drop: [f for f in features if f not in drop]
    return {
        "full_45": ("W1-B-01", features, "Y"),
        "minus_tier1_38": ("W1-B-02", keep(TIER1_LIST), "partial (tier 2 retained)"),
        "minus_family_32": ("W1-B-03", keep(FAMILY_LIST), "N (tiers 1 and 2 removed)"),
        "family_only_13": ("W1-B-04", [f for f in features if f in FAMILY_LIST], "only"),
    }


def evidence(frame, out):
    """Facts behind the tiering: own rating sits inside PR/NR, so PR/NR are not independent of the label rule."""
    d = frame[frame.eligible]
    rows = []
    one = d[d.total_reviews == 1]
    rows.append({"fact": "share of eligible reviews written by reviewers with exactly one review in the data",
                 "value": len(one) / len(d), "n": len(one)})
    rows.append({"fact": "among those, PR equals 1 exactly when the rating is 4 or 5",
                 "value": float((one.PR == (one.rating >= 4)).mean()), "n": len(one)})
    rows.append({"fact": "among those, NR equals 1 exactly when the rating is 1 or 2",
                 "value": float((one.NR == (one.rating <= 2)).mean()), "n": len(one)})
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 10 ** 9)]:
        g = d[d.total_reviews.between(lo, hi)]
        rows.append({"fact": f"fake rate for reviewers with {lo}-{hi if hi < 10 ** 9 else 'more'} reviews", "value": g.label.mean(), "n": len(g)})
    r = pd.DataFrame(rows)
    r.to_csv(out / "rating_family_evidence.csv", index=False)
    print(r.round(4).to_string(index=False))


def fit(features, train, device):
    model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.1, subsample=0.85, colsample_bytree=0.9,
                          tree_method="hist", device=device, n_jobs=2, random_state=SEED, eval_metric="logloss")
    model.fit(train[features].astype("float32"), train.label)
    return model


def scorer(model, features):
    return lambda frame: model.get_booster().predict(DMatrix(frame[features].astype("float32")))


def rating_table(train):
    rate = train.groupby("rating").label.mean()
    return lambda frame: frame.rating.map(rate).to_numpy()


def cross_validate(dev, sets, device):
    folds, oof = [], []
    for outer in range(5):
        val_fold = (outer + 1) % 5
        train = dev[~dev.fold.isin([outer, val_fold])]
        val, test = dev[dev.fold == val_fold], dev[dev.fold == outer]
        models = {name: (lambda f, tr=train: scorer(fit(f, tr, device), f))(features) for name, (_, features, _) in sets.items()}
        models["rating_only_table"] = rating_table(train)
        for name, predict in models.items():
            cut = best_f1_threshold(val.label, predict(val))
            score = predict(test)
            folds.append({"model": name, "fold": outer, **report(test.label, score, cut)})
            oof.append(pd.DataFrame({"model": name, "review_id": test.review_id.to_numpy(), "label": test.label.to_numpy(),
                                     "fold": outer, "score": score, "threshold": cut}))
        print(f"outer fold {outer} done (threshold fold {val_fold})", flush=True)
    folds, oof = pd.DataFrame(folds), pd.concat(oof, ignore_index=True)
    pooled = pd.DataFrame([{"model": name, "fold": "pooled", **report(g.label, g.score, flagged=g.score >= g.threshold)}
                           for name, g in oof.groupby("model", sort=False)])
    folds.to_csv(out_path("ablation_cv_folds.csv"), index=False)
    pooled.to_csv(out_path("ablation_cv_pooled.csv"), index=False)
    return folds, pooled


def out_path(name):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(model, features, name, run_id, threshold, package, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / (name + ".ubj")
    model.get_booster().save_model(weights)
    splits = Path(package) / "data/splits"
    spec = {"model_id": run_id, "features": features, "weights": weights.name,
            "weights_sha256": sha256(weights), "threshold": threshold, "seed": SEED,
            "fit_folds": [1, 2, 3, 4], "validation_fold": 0, "device": device,
            "split_sha256": {p.name: sha256(p) for p in sorted(splits.glob("amazon*.parquet"))},
            "versions": {p: version(p) for p in ("xgboost", "pandas", "scikit-learn", "numpy")}}
    (directory / (name + ".json")).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    reloaded = Booster()
    reloaded.load_model(weights)
    return reloaded


def holdout(dev, test, sets, device, package):
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    seen = test.user_id.isin(set(dev.user_id)).to_numpy()
    cohorts = [("all holdout", np.ones(len(test), bool)), ("holdout users seen in development", seen),
               ("holdout users NOT in development", ~seen)]
    scores, rows, gains = {}, [], []
    models = {}
    for name, (_, features, _) in sets.items():
        model = fit(features, train, device)
        models[name] = scorer(model, features)
        cut = best_f1_threshold(val.label, models[name](val))
        reloaded = freeze(model, features, name, sets[name][0], cut, package, OUT / "models", device)
        score = reloaded.predict(DMatrix(test[features].astype("float32")))
        np.testing.assert_array_equal(score, models[name](test))
        if name == "minus_family_32":
            test[["review_id", "label"]].assign(score=score, flagged=score >= cut).to_parquet(
                out_path("minus_family_32_holdout.parquet"), index=False)
            g = pd.DataFrame({"feature": features})
            for kind in ("gain", "total_gain"):
                g[kind + "_share"] = g.feature.map(model.get_booster().get_score(importance_type=kind)).fillna(0)
                g[kind + "_share"] /= g[kind + "_share"].sum()
            gains.append(g.sort_values("total_gain_share", ascending=False))
    models["rating_only_table"] = rating_table(train)
    for name, predict in models.items():
        cut = best_f1_threshold(val.label, predict(val))
        scores[name] = (predict(test), cut)
        for cohort, mask in cohorts:
            rows.append({"model": name, "cohort": cohort, **report(test.label.to_numpy()[mask], scores[name][0][mask], cut)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("ablation_holdout_cohorts.csv"), index=False)
    gains[0].to_csv(out_path("ablation_gain_without_family.csv"), index=False)
    # Reproduction check against Person A's frozen GPU model (same features, same folds).
    booster = Booster()
    spec = json.loads((Path(package) / "models/person_a/model.json").read_text())
    booster.load_model(Path(package) / "models/person_a" / spec["weights"])
    frozen = booster.predict(DMatrix(test[spec["features"]].astype("float32")))
    check = {"frozen_gpu_ap": average_precision_score(test.label, frozen),
             "cpu_refit_ap": average_precision_score(test.label, scores["full_45"][0]),
             "frozen_gpu_auc": roc_auc_score(test.label, frozen),
             "cpu_refit_auc": roc_auc_score(test.label, scores["full_45"][0])}
    pd.Series(check).to_csv(out_path("ablation_reproduction_check.csv"), header=["value"])
    print("reproduction:", {k: round(v, 4) for k, v in check.items()})
    return scores, cohorts, table


def paired_bootstrap(test, scores, cohorts, rounds=400):
    y = test.label.to_numpy()
    group, products = pd.factorize(test.prod_id, sort=True)
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(len(products), size=len(products)), minlength=len(products))[group] for _ in range(rounds)]
    pairs = [("minus_family_32", "full_45"), ("minus_tier1_38", "full_45"), ("minus_family_32", "rating_only_table"),
             ("full_45", "rating_only_table")]
    rows = []
    for cohort, mask in [c for c in cohorts if c[0] != "holdout users seen in development"]:
        for a, b in pairs:
            sa, sb = scores[a][0][mask], scores[b][0][mask]
            point = average_precision_score(y[mask], sa) - average_precision_score(y[mask], sb)
            deltas = []
            for w in draws:
                w = w[mask]
                if 0 < (w * y[mask]).sum() < w.sum():
                    deltas.append(average_precision_score(y[mask], sa, sample_weight=w) - average_precision_score(y[mask], sb, sample_weight=w))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            rows.append({"cohort": cohort, "comparison": f"{a} minus {b}", "delta_ap": point, "lower_95": lo, "upper_95": hi,
                         "product_clusters": len(products), "draws": len(deltas)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("ablation_paired_bootstrap.csv"), index=False)
    print(table.round(3).to_string(index=False))


LOG_COLUMNS = ["test_set", "fold", "n", "positive", "prevalence", "ap", "roc_auc", "threshold", "precision", "recall", "f1",
               "tn", "fp", "fn", "tp", "pr_auc", "date", "owner", "experiment_id", "train_set", "label_definition",
               "feature_set", "rating_family_included", "split_file", "seed", "notes"]


def log_rows(sets, pooled, table, device):
    ids = {name: (eid, len(features), flag) for name, (eid, features, flag) in sets.items()}
    ids["rating_only_table"] = ("W1-B-05", 1, "rating only")
    rows = []
    for _, r in pooled.iterrows():
        eid, k, flag = ids[r.model]
        rows.append({**r.to_dict(), "test_set": "amazon_user_cv_pooled", "experiment_id": eid, "feature_set": f"{r.model} ({k} features)",
                     "rating_family_included": flag, "notes": "CV protocol of W1-A-01: threshold on next fold; pooled with per-fold cutoffs"})
    for _, r in table.iterrows():
        eid, k, flag = ids[r.model]
        rows.append({**r.to_dict(), "test_set": f"amazon_product_holdout | {r.cohort}", "fold": "frozen-style", "experiment_id": eid,
                     "feature_set": f"{r.model} ({k} features)", "rating_family_included": flag,
                     "notes": "fit on folds 1-4, threshold on fold 0, product holdout"})
    log = pd.DataFrame(rows).assign(date=pd.Timestamp.today().date().isoformat(), owner="B",
        train_set=f"Amazon eligible, product holdout excluded ({device} refit)", label_definition="audited Amazon combined label; fake=1",
        split_file="data/splits/amazon_user_folds.parquet", seed=SEED)
    log.reindex(columns=LOG_COLUMNS).to_csv(out_path("ablation_results_log_rows.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, frame, dev, test = load(args.package)
    json.dump({"tier1": {k: FAMILY[k] for k in TIER1_LIST}, "tier2": {k: FAMILY[k] for k in TIER2_LIST},
               "not_in_family_but_related": NOT_IN_FAMILY_BUT_RELATED, "family": FAMILY_LIST},
              open(out_path("rating_family.json"), "w"), indent=2)
    evidence(frame, OUT)
    sets = feature_sets(spec["features"])
    _, pooled = cross_validate(dev, sets, args.device)
    scores, cohorts, table = holdout(dev, test, sets, args.device, args.package)
    paired_bootstrap(test, scores, cohorts)
    log_rows(sets, pooled, table, args.device)
    print(pooled[["model", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(table[["model", "cohort", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
