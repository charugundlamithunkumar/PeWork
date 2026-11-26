"""Amazon label audit (Workstream B, task 5).

Part 1 (always runs, needs only Person A's package): what the combined label looks like by rating,
reviewer and product, and how many fakes MUST come from the Amazon-removal path.
Part 2 (needs the raw Amazon file, which is not in the package): value counts of every label-defining column,
overlaps, and attribution of each fake review to the rule path or the removal path.

    python -m review_pipeline.person_b_label_audit --package "../Person A"
    python -m review_pipeline.person_b_label_audit --package "../Person A" --raw path/to/amazon_data.csv

Column names are the ones the notebook's build_amazon_label() uses; any that are absent are reported, not assumed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import default_package

OUT = Path(__file__).resolve().parents[1] / "results/person_b"
RULE_COLS = ["fake_review_product", "reviewer_classified_fake"]
REMOVAL_COLS = ["deleted_by_amazon", "review_is_removed_by_amazon"]
OTHER_COLS = ["fake_review_campaign_start_date", "reviewer_classified_real", "reviewer_classified_honest",
              "reviewer_label", "reviewer_labeled_fake", "reviewer_labeled_honest"]


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return df


def from_package(package):
    package = Path(package)
    feats = pd.read_parquet(package / "data/features/amazon_person_a.parquet", columns=["review_id", "user_id", "prod_id", "rating", "label", "total_reviews"])
    split = pd.read_parquet(package / "data/splits/amazon_user_folds.parquet", columns=["review_id", "eligible"])
    d = feats.merge(split, on="review_id", validate="one_to_one")
    e, ie = d[d.eligible], d[~d.eligible]
    rows = [
        ("all rows", len(d), d.label.mean()),
        ("eligible rows (used for training and evaluation)", len(e), e.label.mean()),
        ("ineligible rows (excluded; legacy label shown)", len(ie), ie.label.mean()),
    ]
    summary = save(pd.DataFrame(rows, columns=["group", "rows", "fake_rate"]), "label_audit_overview.csv")
    print(summary.round(4).to_string(index=False))

    by = e.groupby("rating").label.agg(fakes="sum", reviews="size", fake_rate="mean").reset_index()
    by["share_of_all_fakes"] = by.fakes / by.fakes.sum()
    save(by, "label_audit_by_rating.csv")
    print("\nfake labels by star rating (eligible rows)\n", by.round(4).to_string(index=False))
    non5 = int(by.loc[by.rating != 5, "fakes"].sum())
    print(f"\n{non5:,} fake labels ({non5 / by.fakes.sum():.1%} of all fakes) have rating < 5. The rating == 5 rule cannot produce them, "
          f"so they must come from the Amazon-removal path. The share of 5-star fakes that are ALSO removed is not recoverable here (needs --raw).")
    print(f"{by.loc[by.rating == 5, 'fake_rate'].iloc[0]:.1%} of eligible 5-star reviews are labelled fake.")

    ie_by = ie.groupby("rating").label.agg(reviews="size", fake_rate="mean").reset_index()
    save(ie_by, "label_audit_ineligible_by_rating.csv")
    print("\nineligible rows by star rating\n", ie_by.round(4).to_string(index=False))

    multi = e.groupby("user_id").label.agg(["size", "mean"])
    multi = multi[multi["size"] >= 2]
    kind = np.select([multi["mean"] == 0, multi["mean"] == 1], ["all genuine", "all fake"], "mixed")
    reviewer = pd.Series(kind).value_counts().rename_axis("reviewer type").reset_index(name="reviewers")
    reviewer["share"] = reviewer.reviewers / reviewer.reviewers.sum()
    save(reviewer, "label_audit_reviewer_consistency.csv")
    print(f"\nreviewers with 2+ eligible reviews ({len(multi):,}):\n", reviewer.round(4).to_string(index=False))

    prod = e.groupby("prod_id").label.agg(["size", "mean"])
    buckets = pd.cut(prod["mean"], [-0.001, 0, 0.25, 0.75, 1.0], labels=["no fakes", "1-25% fake", "26-75% fake", ">75% fake"])
    pr = prod.groupby(buckets, observed=True).agg(products=("size", "size"), reviews=("size", "sum")).reset_index()
    pr = pr.rename(columns={"mean": "product fake-rate bucket"})
    save(pr, "label_audit_product_buckets.csv")
    print(f"\nproducts ({len(prod):,}) by share of fake labels:\n", pr.to_string(index=False))

    tr = e.assign(bucket=pd.cut(e.total_reviews, [0, 1, 3, 10, 10 ** 9], labels=["1", "2-3", "4-10", "11+"]))
    t = tr.groupby("bucket", observed=True).label.agg(reviews="size", fake_rate="mean").reset_index()
    save(t, "label_audit_by_reviewer_activity.csv")
    print("\nfake rate by the reviewer's review count IN THIS DATASET\n", t.round(4).to_string(index=False))


def from_raw(raw, package):
    from itertools import combinations

    raw = Path(raw)
    fields = RULE_COLS + REMOVAL_COLS + OTHER_COLS
    wanted = {"review_id", "review_rating", "rating", *fields}
    if raw.suffix == ".parquet":
        df = pd.read_parquet(raw)
    else:
        df = pd.read_csv(raw, usecols=lambda col: col.strip().lower() in wanted, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    split = pd.read_parquet(Path(package) / "data/splits/amazon_user_folds.parquet", columns=["review_id", "eligible"])
    labels = pd.read_parquet(Path(package) / "data/features/amazon_person_a.parquet", columns=["review_id", "label"])
    for table in (split, labels):
        df = df.merge(table, on="review_id", how="outer", validate="one_to_one", indicator=True)
        if df.review_id.isna().any() or not df.pop("_merge").eq("both").all():
            raise ValueError("Raw review IDs and the Person A handoff must match exactly.")
    present = [c for c in fields if c in df]
    counts = []
    for col in fields:
        values = df[col].astype("string").fillna("(missing)") if col in present else None
        for value, n in (values.value_counts().items() if values is not None else [("(absent)", len(df))]):
            counts.append({"column": col, "value": value, "rows": n, "share": n / len(df)})
    save(pd.DataFrame(counts), "label_audit_raw_value_counts.csv")

    binary = {"true": True, "1": True, "1.0": True, "yes": True,
              "false": False, "0": False, "0.0": False, "no": False}
    flags = {}
    for col in present:
        if col in ("fake_review_campaign_start_date", "reviewer_label"):
            continue
        text = df[col].astype("string").str.strip().str.lower()
        flags[col] = text.map(binary).astype("boolean")
        if (text.notna() & flags[col].isna()).any():
            raise ValueError(f"Unrecognized flag in {col}")
    aliases = [flags[c] for c in REMOVAL_COLS if c in flags]
    if not aliases:
        raise ValueError("Missing Amazon removal outcome column")
    removed = aliases[0]
    for alias in aliases[1:]:
        if (removed != alias).any():
            raise ValueError("Inconsistent observed Amazon removal aliases")
        removed = removed.fillna(alias)
    stars = df["review_rating" if "review_rating" in df else "rating"].astype("Float64")
    five = stars.eq(5)
    rule = five & flags[RULE_COLS[0]] & flags[RULE_COLS[1]]
    combined = removed | rule
    eligible = df.eligible
    if combined[eligible].isna().any() or not combined[eligible].eq(df.label[eligible]).all():
        raise ValueError("Reconstructed eligible Amazon labels disagree with Person A")
    def state(col):
        return col.astype("string").fillna("unknown")

    paths = pd.DataFrame({"rule": state(rule), "recorded_removal": state(removed), "combined": state(combined)})
    tables = []
    for scope, sub in [("raw", paths), ("eligible", paths[eligible]), ("ineligible", paths[~eligible])]:
        out = sub.value_counts().reset_index(name="rows").assign(scope=scope)
        out["share_of_scope"] = out.rows / len(sub)
        tables.append(out)
    save(pd.concat(tables, ignore_index=True), "label_audit_paths.csv")
    cross = []
    factors = {**flags, "rating_is_five": five, "rule": rule}
    for left, right in combinations(factors, 2):
        pair = pd.DataFrame({"left_value": state(factors[left]), "right_value": state(factors[right])})
        cross.append(pair.value_counts().reset_index(name="rows").assign(left=left, right=right))
    save(pd.concat(cross, ignore_index=True), "label_audit_raw_crosstabs.csv")
    save(pd.crosstab(stars, state(rule)).reset_index(), "label_audit_rating_vs_rule.csv")
    save(pd.crosstab(state(flags[RULE_COLS[0]]), state(flags[RULE_COLS[1]])).reset_index(),
         "label_audit_product_vs_reviewer_flag.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=default_package())
    parser.add_argument("--raw", default=None)
    args = parser.parse_args()
    from_package(args.package)
    if args.raw:
        from_raw(args.raw, args.package)
    else:
        print("\n[part 2 skipped] pass --raw with the original Amazon file to audit the label-defining columns.")


if __name__ == "__main__":
    main()
