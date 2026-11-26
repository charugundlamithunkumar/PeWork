"""Which feature groups help on NEW products? (Workstream B follow-up to the rating-family ablation)

Selection signal: product-grouped five-fold CV INSIDE the development data only.
Confirmation: the product holdout, scored once per feature set and never used to choose anything.

    python -m review_pipeline.person_b_feature_groups --package "../Person A"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import default_package
from .person_b_ablation import (LOG_COLUMNS, SEED, best_f1_threshold, fit, load, out_path, rating_table,
                                report, scorer)
from .rating_family import FAMILY_LIST, GROUPS

PAIRS = [("family_13", "full_45"), ("family_plus_user_behavior_22", "full_45"),
         ("family_plus_text_style_32", "full_45"), ("family_plus_product_context_17", "family_13"),
         ("family_13", "rating_only_table"), ("full_45", "rating_only_table")]


def sets_for(features):
    pick = lambda *parts: [f for f in features if f in set(sum(parts, []))]
    fam = FAMILY_LIST
    return {
        "full_45": features,
        "family_13": pick(fam),
        "family_plus_product_context_17": pick(fam, GROUPS["product_context"]),
        "family_plus_user_behavior_22": pick(fam, GROUPS["user_behavior"]),
        "family_plus_text_style_32": pick(fam, GROUPS["text_style"]),
        "no_family_32": pick(GROUPS["product_context"], GROUPS["user_behavior"], GROUPS["text_style"]),
        "no_family_no_product_context_28": pick(GROUPS["user_behavior"], GROUPS["text_style"]),
        "full_minus_product_context_41": pick(fam, GROUPS["user_behavior"], GROUPS["text_style"]),
    }


def product_cv(dev, sets, device):
    groups, _ = pd.factorize(dev.prod_id, sort=True)
    fold = np.full(len(dev), -1)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, groups=groups)):
        fold[idx] = k
    dev = dev.assign(pfold=fold)
    assert dev.groupby("prod_id").pfold.nunique().max() == 1
    rows, oof = [], []
    for outer in range(5):
        val_fold = (outer + 1) % 5
        train = dev[~dev.pfold.isin([outer, val_fold])]
        val, test = dev[dev.pfold == val_fold], dev[dev.pfold == outer]
        models = {n: scorer(fit(f, train, device), f) for n, f in sets.items()}
        models["rating_only_table"] = rating_table(train)
        for name, predict in models.items():
            cut = best_f1_threshold(val.label, predict(val))
            score = predict(test)
            rows.append({"model": name, "fold": outer, **report(test.label, score, cut)})
            oof.append(pd.DataFrame({"model": name, "label": test.label.to_numpy(), "score": score, "threshold": cut}))
        print(f"product-grouped fold {outer} done", flush=True)
    folds, oof = pd.DataFrame(rows), pd.concat(oof, ignore_index=True)
    pooled = pd.DataFrame([{"model": n, "fold": "pooled", **report(g.label, g.score, flagged=g.score >= g.threshold)}
                           for n, g in oof.groupby("model", sort=False)])
    summary = folds.groupby("model", sort=False).ap.agg(["mean", "std"]).rename(columns={"mean": "fold_mean_ap", "std": "fold_sd_ap"})
    pooled = pooled.merge(summary, left_on="model", right_index=True)
    folds.to_csv(out_path("groups_product_cv_folds.csv"), index=False)
    pooled.to_csv(out_path("groups_product_cv_pooled.csv"), index=False)
    return folds, pooled


def confirm(dev, test, sets, device):
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    seen = test.user_id.isin(set(dev.user_id)).to_numpy()
    cohorts = [("all holdout", np.ones(len(test), bool)), ("holdout users NOT in development", ~seen)]
    models = {n: scorer(fit(f, train, device), f) for n, f in sets.items()}
    models["rating_only_table"] = rating_table(train)
    scores, rows = {}, []
    for name, predict in models.items():
        cut = best_f1_threshold(val.label, predict(val))
        scores[name] = predict(test)
        for cohort, mask in cohorts:
            rows.append({"model": name, "cohort": cohort, **report(test.label.to_numpy()[mask], scores[name][mask], cut)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("groups_holdout_confirmation.csv"), index=False)
    return scores, cohorts, table


def bootstrap(test, scores, cohorts, rounds=400):
    y = test.label.to_numpy()
    group, products = pd.factorize(test.prod_id, sort=True)
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(len(products), size=len(products)), minlength=len(products))[group] for _ in range(rounds)]
    rows = []
    for cohort, mask in cohorts:
        for a, b in PAIRS:
            point = average_precision_score(y[mask], scores[a][mask]) - average_precision_score(y[mask], scores[b][mask])
            deltas = [average_precision_score(y[mask], scores[a][mask], sample_weight=w[mask])
                      - average_precision_score(y[mask], scores[b][mask], sample_weight=w[mask])
                      for w in draws if 0 < (w[mask] * y[mask]).sum() < w[mask].sum()]
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            rows.append({"cohort": cohort, "comparison": f"{a} minus {b}", "delta_ap": point, "lower_95": lo, "upper_95": hi, "draws": len(deltas)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("groups_paired_bootstrap.csv"), index=False)
    print(table.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, _, dev, test = load(args.package)
    sets = sets_for(spec["features"])
    assert sorted(sets["no_family_32"] + FAMILY_LIST) == sorted(spec["features"])
    json.dump({k: v for k, v in sets.items()}, open(out_path("groups_feature_sets.json"), "w"), indent=2)
    _, pooled = product_cv(dev, sets, args.device)
    print(pooled[["model", "ap", "roc_auc", "precision", "recall", "f1", "fold_mean_ap", "fold_sd_ap"]].round(3).to_string(index=False))
    scores, cohorts, table = confirm(dev, test, sets, args.device)
    print(table[["model", "cohort", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))
    bootstrap(test, scores, cohorts)


if __name__ == "__main__":
    main()
