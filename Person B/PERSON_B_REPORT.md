# Person B | Amazon label audit and rating ablation

Updated November 26, 2026. The primary analysis uses the full 673-product holdout. B's CPU refits and A's original GPU baseline have distinct weights. C now loads B's saved 32-feature weights.

## 1. The rating-derived family

| Tier | Features | Why they are rating-derived |
|---|---|---|
| 1: the review's own rating is inside | `rating`, `EXT`, `sentiment_mismatch`, `RD_raw`, `ERD`, `PR`, `NR` | `RD_raw` and `ERD` use the review's rating against the product mean. `PR` and `NR` are shares over the reviewer's **whole** history, including this review. |
| 2: other reviews' ratings shape the value | `burst_ratio`, `is_burst_day`, `pos_burst_ratio`, `is_pos_burst_day`, `negative_review_concentration`, `burst_text_similarity_to_peers` | Burst counts use product-day tallies of 1-2 star or 4-5 star reviews. The last feature is only computed on negative-burst days. |

Kept out of the family on purpose: `vader_compound` (text only), `total_reviews`, `prod_review_count`, `MNR`, `BST` (counts and dates, no ratings). The machine-readable list is `results/person_b/rating_family.json`.

**Evidence that `PR` and `NR` are not independent of the rating:** 76.9% of eligible reviews come from reviewers with exactly one review in this data, and for all of them `PR` is 1 exactly when the rating is 4-5 and `NR` is 1 exactly when it is 1-2 (100%, `rating_family_evidence.csv`).

## 2. Paired ablation

The development set has 232,334 eligible reviews. Reviewer CV uses three folds for fitting, the next fold for threshold selection and the outer fold for testing. For frozen holdout evaluation, folds 1–4 train each model and fold 0 selects its fake-class F1 cutoff. AP is average precision. PR-AUC integrates the precision–recall curve with trapezoids.

| Feature set | Reviewer CV AP | CV ROC-AUC | CV F1 | Holdout AP | AP excluding three products | Same exclusion, new reviewers AP |
|---|---:|---:|---:|---:|---:|---:|
| Full 45 | 0.906 | 0.938 | 0.834 | 0.517 | 0.835 | 0.481 |
| Without tier 1 (38) | 0.871 | 0.922 | 0.796 | 0.485 | 0.799 | 0.367 |
| Without the family (32) | 0.860 | 0.917 | 0.784 | 0.475 | 0.785 | 0.339 |
| Family only (13) | 0.875 | 0.919 | 0.809 | 0.800 | 0.815 | 0.469 |
| Rating only | 0.715 | 0.855 | 0.783 | 0.629 | 0.658 | 0.304 |

On all held-out products, removing the 13 rating-derived inputs reduced AP by 0.042 in the paired CPU comparison (95% product-bootstrap interval for the difference −0.064 to −0.022). The remaining features include whole-dataset user and product aggregates. The ablation measures how much performance survives removing direct and derived rating features under this protocol. [Paired intervals](results/person_b/ablation_paired_bootstrap.csv)

The canonical 32-feature model uses cutoff **0.3535873** on 74,192 holdout reviews, with **5,579 false positives, 5,158 false negatives, 9,808 true positives and 53,647 true negatives**. The saved bundle identifies the split hashes, ordered columns, weights checksum and library versions. C checked every holdout score and decision against B's predictions. [Model manifest](results/person_b/models/minus_family_32.json) · [Holdout evaluations](results/person_b/ablation_holdout_cohorts.csv)

## 3. Raw label audit

All 381,734 raw review IDs matched A's feature and split files one-to-one. Among 306,526 eligible reviews, 72,806 carried the combined positive label (23.75%). The rule requires five stars and both the product and reviewer flags.

| Eligible positive pathway | Reviews |
|---|---:|
| Recorded removal, rule false | 20,597 |
| Recorded removal, rule true | 18,346 |
| Rule true, recorded non-removal | 22,878 |
| Rule true, removal unknown | 10,985 |

Recorded removals account for **38,943 / 72,806 (53.5%)** of eligible positives. The 19,414 positives with fewer than five stars (26.7%) are a subset of these removals. The excluded population contains 75,190 reviews whose combined label is unknown and 18 additional cleaning exclusions, including two legacy positives. All eligible labels matched A's labels after reconstruction. [Raw and eligible counts](results/person_b/label_audit_paths.csv)

The populated removal field is `review_is_removed_by_amazon`, with 86,175 unknown values across the raw file. The alias `deleted_by_amazon` is absent. Each reviewer-labeled field is missing in 301,453 rows. Among reviewers with two or more eligible reviews, 59.7% have exclusively positive labels, 29.0% exclusively negative labels and 11.3% mixed labels. These patterns matter when applying reviewer-level predictors to other populations. [All field values](results/person_b/label_audit_raw_value_counts.csv) · [Cross-tabulations](results/person_b/label_audit_raw_crosstabs.csv)

The authors attribute the reviewer classification to Feldman and colleagues' relational model and the product classification to He and colleagues. Their March 2026 update documents unreliable removal measurement after June 15, 2020. The local counts above describe the project's supplied file. The published data page offers updated CSV and JSON downloads from the same research project. [Authors' dataset and codebook](https://bretthollenbeck.com/fake-reviews-data/)

## Other dataset leads from the earlier investigation

- **The 26.7M-review set** (arXiv 2509.21579) cites Hussain et al., IEEE Access 2020, as its source and does not say how its spam labels were made. Nothing connects it to our He/Hollenbeck data. It is not a usable second labelled set.
- **Incentivized-review data** (arXiv 2102.04217) has product-level labels only and says "we will share" with no link. It is the same platform in any case.
- The Dianping and Taobao leads from earlier are unchanged and still unverified.

## Three-product sensitivity

A's frozen Week 1 model flagged all 4,162 reviews on three products with only 492 combined positives. The earlier-history model flagged 203. Their APs are 0.548 and 0.822 on all products and 0.837 and 0.841 after excluding those three. B selected the products after inspecting errors. In the current CPU probe, removing `prod_review_count` alone changed AP from 0.517 to 0.819 and the flag count from 4,162 to 230. Clipping the count to the development maximum did not change the decisions. The experiment implicates a full-platform predictor and does not establish a causal mechanism. [Product probe](results/person_b/product_context_probe.csv) · [Clipping check](results/person_b/range_clipping_test.csv)

The strongest independent test still requires an unused labeled cohort. Use the preselected earlier-history inputs and the frozen validation cutoff for that evaluation.
