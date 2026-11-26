"""The rating-derived feature family, with the reason each feature is in it.

Tiers describe how a star rating gets into the feature, read from the feature code in
ecommerce_feature_pipeline_GPU_final.ipynb (build_group_a, build_group_d_defamation).

TIER1: the review's OWN star rating is inside the formula or inside the aggregate.
TIER2: other reviews' star ratings (or a rating-gated condition) shape the value.
"""

TIER1 = {
    "rating": "the star rating itself",
    "EXT": "1 if rating is 1 or 5",
    "sentiment_mismatch": "|(rating - 3) / 2 - VADER compound|",
    "RD_raw": "|rating - product mean rating|, the mean includes this review",
    "ERD": "1 if RD_raw >= 4",
    "PR": "share of the user's reviews rated 4-5, over ALL their reviews including this one",
    "NR": "share of the user's reviews rated 1-2, over ALL their reviews including this one",
}

TIER2 = {
    "burst_ratio": "product-day count of 1-2 star reviews over the product's median daily count",
    "is_burst_day": "1 if burst_ratio >= 3",
    "pos_burst_ratio": "product-day count of 4-5 star reviews over the product's median daily count",
    "is_pos_burst_day": "1 if pos_burst_ratio >= 3",
    "negative_review_concentration": "distinct products / count over the user's 1-2 star reviews",
    "burst_text_similarity_to_peers": "text similarity, computed only on negative-burst days (is_burst_day == 1)",
}

# Related to ratings only loosely; NOT in the family and kept in every model.
NOT_IN_FAMILY_BUT_RELATED = {
    "vader_compound": "text sentiment; correlates with stars but is computed from the text only",
    "total_reviews": "count of the user's reviews, no ratings involved",
    "prod_review_count": "count of the product's reviews, no ratings involved",
    "time_since_product_first_review": "dates only; can proxy for a campaign window",
    "MNR": "max reviews per day by the user, no ratings involved",
    "BST": "review within 3 days of the user's previous one, no ratings involved",
}

FAMILY = {**TIER1, **TIER2}
TIER1_LIST = list(TIER1)
TIER2_LIST = list(TIER2)
FAMILY_LIST = TIER1_LIST + TIER2_LIST


# The 32 features outside the rating family, grouped by what they describe (used by person_b_feature_groups).
PRODUCT_CONTEXT = ["prod_review_count", "time_since_product_first_review",
                   "max_text_similarity_same_product", "shingle_jaccard_max_same_product"]
USER_BEHAVIOR = ["MNR", "total_reviews", "BST", "reviews_per_day_rate", "day_of_week_entropy",
                 "posting_gap_fano_factor", "co_reviewer_count", "max_product_overlap_jaccard",
                 "reviewer_clustering_coeff"]
TEXT_STYLE = ["vader_compound", "char_count", "word_count", "avg_word_len", "uppercase_ratio",
              "exclamation_density", "promo_phrase_count", "promo_phrase_density", "first_person_ratio",
              "lexical_diversity", "digit_count", "specificity_proxy", "elongation_count", "slang_count",
              "style_dist_to_own_centroid", "user_style_self_consistency", "readability_score",
              "purchase_declaration_count", "title_body_sentiment_mismatch"]
GROUPS = {"product_context": PRODUCT_CONTEXT, "user_behavior": USER_BEHAVIOR, "text_style": TEXT_STYLE}
