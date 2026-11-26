"""Independent audit of the Person A handoff. Runs from the package folder alone.

Needs only data/, models/ and results/ inside this folder (no parent project).
    python -m review_pipeline.person_a_audit            # CPU, about 2 minutes
    python -m review_pipeline.person_a_audit --device cuda

Writes CSVs to results/person_a/audit/ and the confusion-matrix figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import Booster, DMatrix

from .io import Run, digest
from .models import predict, threshold_for, xgb
from .person_a import evaluate

ROOT = Path(__file__).resolve().parents[1]
PRO = "prospective_20261121T180956177896Z"
SEED = 42


def load():
    spec = json.loads((ROOT / "models/person_a/model.json").read_text())
    feats = pd.read_parquet(ROOT / "data/features/amazon_person_a.parquet")
    folds = pd.read_parquet(ROOT / "data/splits/amazon_user_folds.parquet")
    hold = pd.read_parquet(ROOT / "data/splits/amazon_product_holdout.parquet")
    pd.testing.assert_frame_equal(folds[hold.columns].set_index("review_id").sort_index(),
                                  hold.set_index("review_id").sort_index())
    for platform, ids in [("amazon", folds), ("yelp", pd.read_parquet(ROOT / "data/splits/yelp_user_folds.parquet"))]:
        assert set(ids.fold.unique()) == set(range(5)) and ids.groupby("user_id").fold.nunique().eq(1).all()
        if platform == "yelp":
            yelp = pd.read_parquet(ROOT / "data/features/yelp_person_a.parquet",
                                   columns=["review_id", "user_id", "prod_id", *spec["features"]])
            check = ids.merge(yelp, on=["review_id", "user_id", "prod_id"], validate="one_to_one")
            assert len(check) == len(ids) == len(yelp)
    assert set(spec["features"]).issubset(feats.columns)
    frame = feats.merge(folds, on=["review_id", "user_id", "prod_id", "source_row"], validate="one_to_one")
    assert len(frame) == len(feats) == len(folds) == len(hold)
    dev = frame[frame.eligible & ~frame.product_holdout].reset_index(drop=True)
    test = frame[frame.eligible & frame.product_holdout].reset_index(drop=True)
    return spec, dev, test


def integrity(spec, dev, test, out):
    checks = {
        "feature list has 45 unique names": len(set(spec["features"])) == 45,
        "development rows match model.json fit+validation": len(dev) == spec["fit_rows"] + spec["validation_rows"],
        "holdout rows match model.json": len(test) == spec["product_test_rows"],
        "each development user sits in exactly one fold": dev.groupby("user_id").fold.nunique().max() == 1,
        "development and holdout products are disjoint": set(dev.prod_id).isdisjoint(test.prod_id),
        "labels are 0/1": set(dev.label.unique()) | set(test.label.unique()) == {0, 1},
    }
    overlap = test.user_id.isin(set(dev.user_id))
    table = pd.DataFrame({"check": list(checks), "passed": [bool(v) for v in checks.values()], "kind": "must pass"})
    table.loc[len(table)] = ["reviewer groups verified across both platforms", True, "must pass"]
    table.loc[len(table)] = ["all predictors present in both feature tables", True, "must pass"]
    table.loc[len(table)] = ["development and holdout reviewers are disjoint", not overlap.any(), "information"]
    table.to_csv(out / "integrity_checks.csv", index=False)
    assert table[table.kind == "must pass"].passed.all(), "an integrity check failed"
    print(table.to_string(index=False))
    print(f"\nholdout rows from users also in development: {overlap.mean():.1%} "
          f"({int(overlap.sum()):,} rows, prevalence {test.label[overlap].mean():.1%})")
    return overlap.to_numpy()


def cohort_table(spec, dev, test, seen, out):
    booster = Booster()
    weights = ROOT / "models/person_a" / spec["weights"]
    assert digest(weights) == spec["weights_sha256"]
    booster.load_model(weights)
    week1 = booster.predict(DMatrix(test[spec["features"]].astype("float32")))
    logged = pd.read_csv(ROOT / "results/person_a/results_log.csv")
    logged = logged[logged.test_set == "amazon_product_holdout"].iloc[0]
    actual = evaluate(test.label, week1, spec["threshold"])
    for field in actual:
        np.testing.assert_allclose(actual[field], float(logged[field]), rtol=0, atol=1e-6, err_msg=field)
    print("\nFrozen model reproduces every holdout metric in results_log.csv.")

    pro = pd.read_parquet(ROOT / "results/person_a" / PRO / "product_holdout_predictions.parquet")
    pro = test[["review_id", "rating", "prod_id"]].merge(pro, on=["review_id", "prod_id"], validate="one_to_one")
    assert len(pro) == len(test) and (pro.label.to_numpy() == test.label.to_numpy()).all()
    pro_spec = json.loads((ROOT / "results/person_a" / PRO / "model.json").read_text())

    train = dev[dev.fold != 0]
    rate = train.groupby("rating").label.mean()
    val = dev[dev.fold == 0]
    rating_cut = threshold_for(val.label, val.rating.map(rate))
    scores = {"week1_45_features": (week1, spec["threshold"]),
              "history_tree_follow_up": (pro.selected.to_numpy(), pro_spec["threshold"]),
              "rating_only": (test.rating.map(rate).to_numpy(), rating_cut)}
    rows = []
    for cohort, mask in [("all holdout", np.ones(len(test), bool)), ("holdout users seen in development", seen),
                         ("holdout users NOT in development", ~seen)]:
        for name, (score, cut) in scores.items():
            rows.append({"cohort": cohort, "model": name, **evaluate(test.label.to_numpy()[mask], score[mask], cut)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "cohort_metrics.csv", index=False)
    print("\n", table[["cohort", "model", "n", "prevalence", "ap", "roc_auc", "precision", "recall", "f1"]]
          .round(3).to_string(index=False))
    return scores


def calibration(scores, y, out):
    rows = []
    for name in ["week1_45_features", "history_tree_follow_up"]:
        score = scores[name][0]
        bins = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
        for b in range(10):
            m = bins == b
            rows.append({"model": name, "decile": b + 1, "n": int(m.sum()), "mean_score": score[m].mean(),
                         "observed_fake_rate": y[m].mean()})
    table = pd.DataFrame(rows)
    brier = pd.DataFrame([{"model": n, "brier": float(np.mean((s - y) ** 2))} for n, (s, _) in scores.items()]
                         + [{"model": "constant_holdout_prevalence", "brier": float(np.mean((y.mean() - y) ** 2))}])
    table.to_csv(out / "calibration_deciles.csv", index=False)
    brier.to_csv(out / "brier.csv", index=False)
    print("\n", brier.round(4).to_string(index=False))
    print(table[table.decile >= 9].round(3).to_string(index=False))


def paired_bootstrap(test, scores, seen, out, rounds=400):
    """Resample whole products, like the original paired interval, for several cohorts."""
    y = test.label.to_numpy()
    group, products = pd.factorize(test.prod_id, sort=True)
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(len(products), size=len(products)), minlength=len(products))[group]
             for _ in range(rounds)]
    pairs = [("history_tree_follow_up", "week1_45_features"), ("history_tree_follow_up", "rating_only"),
             ("rating_only", "week1_45_features")]
    rows = []
    for cohort, mask in [("all holdout", np.ones(len(y), bool)), ("holdout users NOT in development", ~seen)]:
        for a, b in pairs:
            point = (average_precision_score(y[mask], scores[a][0][mask])
                     - average_precision_score(y[mask], scores[b][0][mask]))
            deltas = []
            for w in draws:
                w = w[mask]
                if 0 < (w * y[mask]).sum() < w.sum():
                    deltas.append(average_precision_score(y[mask], scores[a][0][mask], sample_weight=w)
                                  - average_precision_score(y[mask], scores[b][0][mask], sample_weight=w))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            rows.append({"cohort": cohort, "comparison": f"{a} minus {b}", "metric": "average_precision",
                         "delta": point, "lower_95": lo, "upper_95": hi, "draws": len(deltas)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "paired_bootstrap_ap.csv", index=False)
    print("\n", table.round(3).to_string(index=False))


def strict_retrain(spec, dev, test, seen, device, out):
    """Retrain with every development user who also appears in the holdout removed."""
    features = spec["features"]
    rows = []
    holdout_users = set(test.user_id)
    run = Run(ROOT, "person_a_audit", device=device, seed=SEED)
    for name, pool in [("standard_retrain", dev),
                       ("strict_user_disjoint_retrain", dev[~dev.user_id.isin(holdout_users)])]:
        train, val = pool[pool.fold != 0], pool[pool.fold == 0]
        model = xgb(run).fit(train[features].astype("float32"), train.label)
        actual_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual_device.startswith(device):
            raise RuntimeError(f"Requested {device}, but XGBoost fit on {actual_device}.")
        cut = threshold_for(val.label, predict(model, val, "xgb", features))
        score = predict(model, test, "xgb", features)
        for cohort, mask in [("all holdout", np.ones(len(test), bool)), ("holdout users seen in development", seen),
                             ("holdout users NOT in development", ~seen)]:
            rows.append({"model": name, "cohort": cohort, "train_rows": len(train), "validation_rows": len(val),
                         "device": actual_device, **evaluate(test.label.to_numpy()[mask], score[mask], cut)})
        print(f"{name}: trained on {len(train):,} rows, validation on {len(val):,}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "strict_retrain.csv", index=False)
    print("\n", table[["model", "cohort", "train_rows", "ap", "roc_auc", "precision", "recall", "f1"]]
          .round(3).to_string(index=False))


LOG_COLUMNS = ["test_set", "fold", "n", "positive", "prevalence", "ap", "roc_auc", "threshold", "precision", "recall",
               "f1", "tn", "fp", "fn", "tp", "pr_auc", "date", "owner", "experiment_id", "train_set",
               "label_definition", "feature_set", "rating_family_included", "split_file", "seed", "notes"]


def write_log_rows(out):
    """Rows in results_log.csv format. Regenerated on every run; concatenate, do not edit by hand."""
    cohorts = pd.read_csv(out / "cohort_metrics.csv")
    strict = pd.read_csv(out / "strict_retrain.csv")
    spec = {
        "week1_45_features": ("W1-A-01c", "frozen W1-A-01 model, holdout split by reviewer overlap", "45 shared notebook predictors", "Y"),
        "rating_only": ("W1-A-02", "star-rating rate table fit on development folds 1-4", "rating only", "Y"),
        "history_tree_follow_up": ("W1-A-03", "exploratory follow-up, reviewed on previously examined holdout",
                                   "18 current-review + 7 earlier-product features", "Y"),
    }
    rows = []
    for _, r in cohorts.iterrows():
        eid, note, feats, rated = spec[r.model]
        rows.append({**r.to_dict(), "test_set": f"amazon_product_holdout | {r.cohort}", "fold": "frozen",
                     "experiment_id": eid, "feature_set": feats, "rating_family_included": rated,
                     "train_set": "Amazon eligible, product holdout excluded", "notes": note})
    for _, r in strict.iterrows():
        strict_flag = r.model.startswith("strict")
        rows.append({**r.to_dict(), "test_set": f"amazon_product_holdout | {r.cohort}", "fold": "frozen",
                     "experiment_id": "W1-A-04" if strict_flag else "W1-A-04ref",
                     "feature_set": "45 shared notebook predictors", "rating_family_included": "Y",
                     "train_set": ("development minus every reviewer who also appears in the holdout"
                                   if strict_flag else "Amazon eligible, product holdout excluded"),
                     "notes": f"{r.device} refit with W1-A-01 settings; retrospective features retained"})
    log = pd.DataFrame(rows).assign(date=pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat(), owner="A (review)",
        label_definition="audited Amazon combined label; fake=1", split_file="data/splits/amazon_product_holdout.parquet", seed=SEED)
    log.reindex(columns=LOG_COLUMNS).to_csv(out / "results_log_audit_rows.csv", index=False)


def confusion_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = pd.read_csv(ROOT / "results/person_a/results_log.csv")
    pooled = log[log.test_set == "amazon_user_cv_pooled"].iloc[0]
    cohorts = pd.read_csv(out / "cohort_metrics.csv")
    pick = lambda c: cohorts[(cohorts.model == "week1_45_features") & (cohorts.cohort == c)].iloc[0]
    panels = [("Reviewer-grouped CV (pooled)\nproducts shared with training", pooled),
              ("Product holdout, all reviews", pick("all holdout")),
              ("Product holdout, new reviewers only", pick("holdout users NOT in development"))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, r) in zip(axes, panels):
        counts = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(counts[i, j]):,}\n{share[i, j]:.0%} of row", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["negative", "positive"])
        ax.set_yticks([0, 1], ["negative", "positive"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(f"{title}\nAP {r.ap:.2f} | precision {r.precision:.0%} | recall {r.recall:.0%}", fontsize=9)
    fig.suptitle("Week 1 Amazon XGBoost · Confusion matrices at validation cutoffs", fontsize=11)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/week1_confusion_matrices.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    out = ROOT / "results/person_a/audit"
    out.mkdir(parents=True, exist_ok=True)
    spec, dev, test = load()
    seen = integrity(spec, dev, test, out)
    scores = cohort_table(spec, dev, test, seen, out)
    calibration(scores, test.label.to_numpy(), out)
    paired_bootstrap(test, scores, seen, out)
    strict_retrain(spec, dev, test, seen, args.device, out)
    write_log_rows(out)
    confusion_figure(out)


if __name__ == "__main__":
    main()
