"""Do a few products drive the holdout gap between the 13-feature family model and the full 45-feature model?

    python -m review_pipeline.person_b_holdout_influence --package "../Person A"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import default_package
from .person_b_ablation import best_f1_threshold, fit, load, out_path, scorer
from .person_b_feature_groups import sets_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, _, dev, test = load(args.package)
    sets = sets_for(spec["features"])
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    out = {}
    for name in ("full_45", "family_13"):
        predict = scorer(fit(sets[name], train, args.device), sets[name])
        cut = best_f1_threshold(val.label, predict(val))
        out[name] = (predict(test), cut)
    t = test[["prod_id", "label"]].copy()
    for name, (score, cut) in out.items():
        t[name + "_flag"] = score >= cut
        t[name + "_fp"] = t[name + "_flag"] & (t.label == 0)
    g = t.groupby("prod_id").agg(reviews=("label", "size"), fakes=("label", "sum"), full_fp=("full_45_fp", "sum"), family_fp=("family_13_fp", "sum"))
    g["extra_fp_full_minus_family"] = g.full_fp - g.family_fp
    g = g.sort_values("extra_fp_full_minus_family", ascending=False)
    g.head(25).reset_index().to_csv(out_path("holdout_product_influence_top25.csv"), index=False)
    total_extra = g.extra_fp_full_minus_family.clip(lower=0).sum()
    print(f"extra false positives of the full model over the family model, summed over products where it is worse: {int(total_extra):,}")
    for k in (5, 10, 25, 50):
        print(f"top {k:>2} products account for {g.extra_fp_full_minus_family.head(k).sum() / total_extra:.1%}")
    print(g.head(8).to_string())
    rows = []
    for k in (0, 5, 10, 25):
        drop = set(g.head(k).index)
        m = ~test.prod_id.isin(drop).to_numpy()
        rows.append({"products_removed": k, "reviews_left": int(m.sum()),
                     "ap_full_45": average_precision_score(test.label[m], out["full_45"][0][m]),
                     "ap_family_13": average_precision_score(test.label[m], out["family_13"][0][m])})
    r = pd.DataFrame(rows)
    r["gap"] = r.ap_family_13 - r.ap_full_45
    r.to_csv(out_path("holdout_product_influence_removal.csv"), index=False)
    print(r.round(3).to_string(index=False))
    print("(Products were ranked on the holdout itself, so this shows concentration, not a selection method.)")


if __name__ == "__main__":
    main()
