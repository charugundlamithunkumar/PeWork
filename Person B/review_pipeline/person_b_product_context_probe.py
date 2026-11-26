"""Which single product-context feature makes a model blanket-flag three huge holdout products?

Adds each of the four product-context features to the 13-feature rating family, one at a time, and also removes
each from the full 45. Attribution only: results here are NOT used to choose the final feature set.

    python -m review_pipeline.person_b_product_context_probe --package "../Person A"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import default_package
from .person_b_ablation import best_f1_threshold, fit, load, out_path, scorer
from .rating_family import FAMILY_LIST, GROUPS

TOP3 = ["B01LSUQSB0", "B076F5P28T", "B07B7K7N3P"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, _, dev, test = load(args.package)
    feats = spec["features"]
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    big = test.prod_id.isin(TOP3).to_numpy()
    sets = {"family_13 (reference)": FAMILY_LIST, "full_45 (reference)": feats}
    for f in GROUPS["product_context"]:
        sets[f"family_13 + {f}"] = [c for c in feats if c in FAMILY_LIST or c == f]
        sets[f"full_45 minus {f}"] = [c for c in feats if c != f]
    rows = []
    for name, columns in sets.items():
        predict = scorer(fit(columns, train, args.device), columns)
        cut = best_f1_threshold(val.label, predict(val))
        score = predict(test)
        rows.append({"feature_set": name, "n_features": len(columns), "holdout_ap": average_precision_score(test.label, score),
                     "flagged_of_4162_in_3_big_products": int((score[big] >= cut).sum()),
                     "ap_without_3_products": average_precision_score(test.label[~big], score[~big])})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("product_context_probe.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
