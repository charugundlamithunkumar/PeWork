"""Rating-family ablation on the product holdout with and without three products that dominate it.

The three products (B01LSUQSB0, B076F5P28T, B07B7K7N3P) were identified AFTER seeing the failure, so this is a
sensitivity check, not a headline. Same protocol as person_b_ablation (fit on folds 1-4, threshold on fold 0).

    python -m review_pipeline.person_b_trimmed_ablation --package "../Person A"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import default_package
from .person_b_ablation import best_f1_threshold, feature_sets, fit, load, out_path, rating_table, report, scorer

TOP3 = ["B01LSUQSB0", "B076F5P28T", "B07B7K7N3P"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec, _, dev, test = load(args.package)
    sets = feature_sets(spec["features"])
    train, val = dev[dev.fold != 0], dev[dev.fold == 0]
    keep = ~test.prod_id.isin(TOP3).to_numpy()
    seen = test.user_id.isin(set(dev.user_id)).to_numpy()
    models = {n: scorer(fit(f, train, args.device), f) for n, (_, f, _) in sets.items()}
    models["rating_only_table"] = rating_table(train)
    rows = []
    for name, predict in models.items():
        cut = best_f1_threshold(val.label, predict(val))
        score = predict(test)
        for cohort, mask in [("all 673 products", np.ones(len(test), bool)), ("without the 3 products", keep),
                             ("without the 3 products, new reviewers only", keep & ~seen)]:
            rows.append({"model": name, "holdout": cohort, **report(test.label.to_numpy()[mask], score[mask], cut)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("ablation_holdout_trimmed.csv"), index=False)
    print(table[["model", "holdout", "n", "ap", "roc_auc", "precision", "recall", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
