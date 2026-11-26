"""Shared single-review extraction and earlier-date platform history."""

from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

EXTRACTOR_VERSION = "text-rating-v1"
PROMO = ("highly recommend", "must buy", "must have", "life changing", "changed my life",
         "best purchase", "exceeded my expectations", "five stars", "worth every penny",
         "will definitely buy again", "buy it now", "don't hesitate", "love love love",
         "amazing product", "great quality", "highly recommended", "works great")
PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our"}
SLANG = {"u", "ur", "gr8", "thx", "pls", "tbh", "imo", "omg", "lol"}
PURCHASE = ("i bought this", "i purchased this", "actually bought", "not sponsored", "honest review", "genuine review")

# Each tuple gives the input mode, whether ratings enter the formula, and its unit.
REGISTRY = {
    "char_count": ("app", False, "characters in review"),
    "word_count": ("app", False, "whitespace-separated words"),
    "avg_word_len": ("app", False, "characters / whitespace-separated words"),
    "uppercase_ratio": ("app", False, "uppercase characters / characters"),
    "exclamation_density": ("app", False, "exclamation marks / characters"),
    "vader_compound": ("app", False, "VADER sentiment compound"),
    "promo_phrase_density": ("app", False, "promotion phrase matches / words"),
    "first_person_ratio": ("app", False, "first person pronouns / regex tokens"),
    "lexical_diversity": ("app", False, "distinct lower-case regex tokens / tokens"),
    "digit_count": ("app", False, "digit characters"),
    "specificity_proxy": ("app", False, "digit characters / characters"),
    "elongation_count": ("app", False, "runs of three or more identical characters"),
    "slang_count": ("app", False, "tokens matching slang vocabulary"),
    "purchase_declaration_count": ("app", False, "purchase declaration phrase matches"),
    "readability_score": ("app", False, "estimated Flesch reading ease"),
    "rating": ("app", True, "stars on a 1 to 5 scale"),
    "EXT": ("app", True, "one or five stars"),
    "sentiment_mismatch": ("app", True, "distance from normalized rating to VADER sentiment"),
    "total_reviews": ("context", False, "earlier reviews by user"),
    "prod_review_count": ("context", False, "earlier reviews for product"),
    "MNR": ("context", False, "maximum earlier daily review count for user"),
    "BST": ("context", False, "prior user review within three days"),
    "time_since_product_first_review": ("context", False, "days since first observed product review"),
    "PR": ("context", True, "share of earlier user reviews rated 4 or 5"),
    "NR": ("context", True, "share of earlier user reviews rated 1 or 2"),
    "RD_raw": ("context", True, "absolute difference from earlier mean product rating"),
    "ERD": ("context", True, "absolute difference at least four stars"),
}
DEFERRED = {
    "burst_ratio", "is_burst_day", "pos_burst_ratio", "is_pos_burst_day",
    "negative_review_concentration", "burst_text_similarity_to_peers",
    "reviews_per_day_rate", "max_text_similarity_same_product", "style_dist_to_own_centroid",
    "user_style_self_consistency", "co_reviewer_count", "max_product_overlap_jaccard",
    "reviewer_clustering_coeff", "day_of_week_entropy", "posting_gap_fano_factor",
    "shingle_jaccard_max_same_product", "title_body_sentiment_mismatch",
}
DEFERRED_RATING = {
    "burst_ratio", "is_burst_day", "pos_burst_ratio", "is_pos_burst_day",
    "negative_review_concentration", "burst_text_similarity_to_peers",
}


@lru_cache(maxsize=1)
def analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def extract_review(text, rating):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter review text.")
    if rating not in range(1, 6):
        raise ValueError("Rating must be a whole number from 1 through 5.")
    lower = text.lower()
    tokens = re.findall(r"\b\w+\b", lower)
    n_words = max(len(text.split()), 1)
    n_chars = len(text)
    sentiment = analyzer().polarity_scores(text)["compound"]
    digits = sum(c.isdigit() for c in text)
    sentences = max(sum(bool(s.strip()) for s in re.split(r"[.!?]+", text)), 1)
    syllables = sum(max(len(re.findall(r"[aeiouy]+", w)) - int(w.endswith("e")), 1) for w in tokens)
    return {
        "char_count": n_chars, "word_count": n_words, "avg_word_len": n_chars / n_words,
        "uppercase_ratio": sum(c.isupper() for c in text) / n_chars,
        "exclamation_density": text.count("!") / n_chars, "vader_compound": sentiment,
        "promo_phrase_density": sum(lower.count(p) for p in PROMO) / n_words,
        "first_person_ratio": sum(w in PRONOUNS for w in tokens) / max(len(tokens), 1),
        "lexical_diversity": len(set(tokens)) / max(len(tokens), 1), "digit_count": digits,
        "specificity_proxy": digits / n_chars, "elongation_count": len(re.findall(r"(.)\1{2,}", text)),
        "slang_count": sum(w in SLANG for w in tokens),
        "purchase_declaration_count": sum(lower.count(p) for p in PURCHASE),
        "readability_score": 206.835 - 1.015 * n_words / sentences - 84.6 * syllables / max(len(tokens), 1),
        "rating": rating, "EXT": int(rating in (1, 5)),
        "sentiment_mismatch": abs((rating - 3) / 2 - sentiment),
    }


def feature_names(mode, ratings):
    return [name for name, (scope, rated, _) in REGISTRY.items()
            if (scope == "app" or mode == "context") and (ratings or not rated)]


def daily_history(frame, key):
    work = frame[[key, "date", "rating"]].assign(
        positive=frame.rating.ge(4).astype(int), negative=frame.rating.le(2).astype(int))
    daily = work.groupby([key, "date"], sort=True, observed=True).agg(
        n=("rating", "size"), stars=("rating", "sum"), pos=("positive", "sum"), neg=("negative", "sum")).reset_index()
    by_group = daily.groupby(key, sort=False, observed=True)
    for col in ("n", "stars", "pos", "neg"):
        daily[f"prior_{col}"] = by_group[col].cumsum() - daily[col]
    daily["day_max"] = by_group.n.cummax()
    daily["prior_max"] = daily.groupby(key, sort=False, observed=True).day_max.shift()
    daily["last_date"] = by_group.date.shift()
    daily["first_date"] = by_group.date.transform("min").where(daily.prior_n.gt(0))
    return frame[[key, "date"]].merge(daily, on=[key, "date"], how="left", validate="many_to_one")


def context_features(frame):
    user = daily_history(frame, "user_id")
    product = daily_history(frame, "prod_id")
    deviation = (frame.rating.reset_index(drop=True) - product.prior_stars / product.prior_n.replace(0, np.nan)).abs()
    return pd.DataFrame({
        "total_reviews": user.prior_n, "prod_review_count": product.prior_n, "MNR": user.prior_max,
        "BST": (user.date - user.last_date).dt.days.le(3).astype(float).where(user.prior_n.gt(0)),
        "time_since_product_first_review": (product.date - product.first_date).dt.days,
        "PR": user.prior_pos / user.prior_n.replace(0, np.nan),
        "NR": user.prior_neg / user.prior_n.replace(0, np.nan),
        "RD_raw": deviation, "ERD": deviation.ge(4).astype(float).where(deviation.notna()),
    }, dtype=float)


def feature_table(frame):
    raw = frame.reset_index(drop=True)
    text = pd.DataFrame.from_records(extract_review(t, r) for t, r in zip(raw.review_text, raw.rating))
    return pd.concat([raw, text.drop(columns="rating"), context_features(raw)], axis=1)


def grouped_folds(frame, n, seed):
    for offset in range(12):
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed + offset)
        pairs = list(splitter.split(frame, frame.label, frame.user_id))
        if len(pairs) == n and all(frame.label.iloc[i].nunique() == 2 and frame.label.iloc[j].nunique() == 2
                                   for i, j in pairs):
            return pairs
    raise ValueError("Could not create grouped folds with both classes. Check reviewer and class counts.")


def build_features(run):
    amz = run.read_table("amazon_reviews.parquet").merge(
        run.read_table("split_manifest.parquet"), on="review_id", validate="one_to_one")
    amz["fold"] = -1
    dev = amz.loc[amz.partition.eq("development")]
    for number, (_, validation) in enumerate(grouped_folds(dev, run.folds, run.seed)):
        amz.loc[dev.index[validation], "fold"] = number
    run.write_table("development_folds.parquet", amz.loc[amz.fold.ge(0), ["review_id", "fold"]])
    # Each grouped development fold has its own earlier history. Inner CV keeps blocks intact.
    amz = pd.concat([feature_table(part) for _, part in amz.groupby(["partition", "fold"])], ignore_index=True)
    yelp = feature_table(run.read_table("yelp_reviews.parquet"))
    for platform, frame in (("amazon", amz), ("yelp", yelp)):
        assert frame.review_id.is_unique
        run.write_table(f"{platform}_features.parquet", frame)
    rows = []
    for name, (scope, rated, formula) in REGISTRY.items():
        rows.append({"feature": name, "mode": scope, "rating_derived": rated, "formula": formula,
                     "available_at": "review time" if scope == "app" else "strictly earlier dates within history block",
                     "missing_policy": "XGBoost missing branch or MCDM training median",
                     "amazon_coverage": float(amz[name].notna().mean()),
                     "yelp_coverage": float(yelp[name].notna().mean())})
    rows.extend({"feature": name, "mode": "deferred", "rating_derived": name in DEFERRED_RATING,
                 "missing_policy": "excluded from v2"} for name in sorted(DEFERRED))
    run.write_table("feature_availability.csv", pd.DataFrame(rows))
    run.write_json("feature_registry.json", {"extractor": EXTRACTOR_VERSION,
                                             "active": {name: list(spec) for name, spec in REGISTRY.items()},
                                             "deferred": sorted(DEFERRED), "excluded": ["RD_robust", "DEV", "WRD", "promo_phrase_count"]})
    sample = amz.iloc[:12].copy().sort_values("date")
    probe = sample.iloc[[0]].copy()
    extra = probe.copy()
    extra["date"] += pd.Timedelta(days=365)
    extra["rating"] = 1
    extra["label"] = 1 - extra["label"]
    pd.testing.assert_frame_equal(context_features(probe), context_features(pd.concat([probe, extra])).iloc[:1])
    relabeled = sample.assign(label=1 - sample.label)
    pd.testing.assert_frame_equal(context_features(sample), context_features(relabeled))
