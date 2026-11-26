"""Build the canonical team log from owner outputs. Run after run_transfer."""

import numpy as np
import pandas as pd

from .run_transfer import A, B, C, ROOT, HISTORY, ablation, read_spec


def build():
    a = pd.read_csv(A / "results/person_a/results_log.csv")
    audit = pd.read_csv(A / "results/person_a/audit/results_log_audit_rows.csv")
    b = pd.read_csv(B / "ablation_results_log_rows.csv")
    legacy = pd.read_csv(B.parent / "legacy_b_gpu32.csv")
    c = pd.read_csv(C / "transfer_results.csv").drop(columns="positives")
    c["experiment_id"] = "C-transfer-" + c.model
    c["feature_set"] = c.model
    c["train_set"] = "Amazon development folds 1-4, cutoff on fold 0"
    c["fold"] = "frozen"
    c["label_definition"] = np.where(c.test_set.str.startswith("yelp"),
                                    "Yelp filtered review=1", "audited Amazon combined label; fake=1")
    c["rating_family_included"] = c.rating_family_included.map({True: "Y", False: "N"})
    for table in (a, audit, b):
        table["split_file"] = A.relative_to(ROOT).as_posix() + "/" + table.split_file
    for table in (a, audit, b, legacy):
        table["model_id"] = table.experiment_id
    audit.loc[audit.experiment_id.eq("W1-A-01c"), "model_id"] = "W1-A-01"
    for path in sorted((B / "models").glob("*.json")):
        spec = read_spec(path)
        mask = b.experiment_id.eq(spec["model_id"]) & b.fold.eq("frozen-style")
        b.loc[mask, "artifact"] = (path.parent / spec["weights"]).relative_to(ROOT).as_posix()
        b.loc[mask, "weights_sha256"] = spec["weights_sha256"]
    rating_path = C / "models/rating_only_table.json"
    for table, mask, path in [
        (a, a.fold.eq("frozen"), A / "models/person_a/amazon_xgb.ubj"),
        (audit, audit.model_id.eq("W1-A-01"), A / "models/person_a/amazon_xgb.ubj"),
        (audit, audit.model_id.eq("W1-A-03"), HISTORY / "model.joblib"),
        (b, b.experiment_id.eq("W1-B-05") & b.fold.eq("frozen-style"), rating_path),
    ]:
        table.loc[mask, "artifact"] = path.relative_to(ROOT).as_posix()
        table.loc[mask, "weights_sha256"] = ablation.sha256(path)
    b["rating_family_scope"] = b.experiment_id.map({
        "W1-B-01": "full", "W1-B-02": "partial", "W1-B-03": "removed",
        "W1-B-04": "family only", "W1-B-05": "rating only"})
    b["rating_family_included"] = np.where(b.rating_family_scope.eq("removed"), "N", "Y")
    log = pd.concat([a, audit, legacy, b, c], ignore_index=True)
    if log.duplicated(["experiment_id", "test_set", "fold"]).any():
        raise ValueError("Duplicate evaluation key")
    if not log.n.eq(log[["tn", "fp", "fn", "tp"]].sum(axis=1)).all():
        raise ValueError("Confusion counts do not sum to cohort size")
    for relative in log.split_file.unique():
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    for path, expected in log[["artifact", "weights_sha256"]].dropna().drop_duplicates().itertuples(index=False):
        if ablation.sha256(ROOT / path) != expected:
            raise ValueError(f"Inconsistent model checksum for {path}")
    outputs = [B.parent / "results_log_team.csv", C / "results_log_team_with_C.csv"]
    for path in outputs:
        log.to_csv(path, index=False)
    print(f"{len(log)} evaluations. Verified identical log copies: {len({ablation.sha256(p) for p in outputs}) == 1}")


if __name__ == "__main__":
    build()
