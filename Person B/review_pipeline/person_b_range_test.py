"""Controlled test: is the blanket-flagging of three huge holdout products caused by prod_review_count being out of range?

Rescores Person A's frozen model on the product holdout twice: as-is, and with prod_review_count clipped to the largest
value seen in development. Nothing is refit and no threshold changes.

    python -m review_pipeline.person_b_range_test --package "../Person A"
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import Booster, DMatrix

from . import default_package
from .person_b_ablation import load, out_path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    args = parser.parse_args()
    package = Path(args.package)
    spec, _, dev, test = load(package)
    booster = Booster()
    booster.load_model(package / "models/person_a" / spec["weights"])
    cap = float(dev.prod_review_count.max())
    over = test.prod_review_count > cap
    print(f"development maximum prod_review_count = {cap:.0f}; holdout rows above it: {int(over.sum()):,} "
          f"in {test.prod_id[over].nunique()} products")
    rows = []
    for name, frame in [("as scored by Week 1", test), ("prod_review_count clipped to development maximum", test.assign(prod_review_count=test.prod_review_count.clip(upper=cap)))]:
        score = booster.predict(DMatrix(frame[spec["features"]].astype("float32")))
        flagged = score >= spec["threshold"]
        rows.append({"scoring": name, "reviews_above_range": int(over.sum()), "flagged_among_those": int(flagged[over.to_numpy()].sum()),
                     "fakes_among_those": int(test.label[over].sum()), **report(test.label, score, spec["threshold"])})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("range_clipping_test.csv"), index=False)
    print(table[["scoring", "reviews_above_range", "flagged_among_those", "fakes_among_those", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
