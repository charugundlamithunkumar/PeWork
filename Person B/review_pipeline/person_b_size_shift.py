"""Development-only shift test: train on smaller products, test on the largest ones.

The product holdout has larger and older products than development (median 386 against 262 reviews per product).
A random product split inside development cannot show what that does to product-volume features, so this script
recreates the shift without using the holdout: products are ranked by review count, the smallest 60% train,
the next 15% choose the threshold, and the largest 25% test.

    python -m review_pipeline.person_b_size_shift --package "../Person A"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import default_package
from .person_b_ablation import SEED, best_f1_threshold, fit, load, out_path, rating_table, report, scorer
from .person_b_feature_groups import sets_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, _, dev, _ = load(args.package)
    sets = sets_for(spec["features"])
    size = dev.groupby("prod_id").size().sort_values(kind="stable")
    order = np.array(size.index)
    n = len(order)
    train_p, val_p, test_p = order[: int(.60 * n)], order[int(.60 * n): int(.75 * n)], order[int(.75 * n):]
    train, val, test = (dev[dev.prod_id.isin(p)] for p in (train_p, val_p, test_p))
    assert not (set(train_p) & set(test_p)) and not (set(val_p) & set(test_p))
    print(f"products train {len(train_p)} (<= {int(size[train_p].max())} reviews), validation {len(val_p)}, "
          f"test {len(test_p)} (>= {int(size[test_p].min())} reviews); rows {len(train):,}/{len(val):,}/{len(test):,}")
    rows = []
    models = {name: scorer(fit(f, train, args.device), f) for name, f in sets.items()}
    models["rating_only_table"] = rating_table(train)
    for name, predict in models.items():
        cut = best_f1_threshold(val.label, predict(val))
        rows.append({"model": name, **report(test.label, predict(test), cut)})
    table = pd.DataFrame(rows).assign(train_rows=len(train), test_products=len(test_p))
    table.to_csv(out_path("size_shift_test.csv"), index=False)
    print(table[["model", "n", "prevalence", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
