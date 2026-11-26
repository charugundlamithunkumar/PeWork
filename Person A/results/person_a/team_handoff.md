# Person A handoff

The [split files](../../data/splits), [feature tables](../../data/features), [frozen GPU model](../../models/person_a/amazon_xgb.ubj), [model settings](../../models/person_a/model.json), and [metrics log](results_log.csv) belong to experiment W1-A-01. The [notebook](../../notebooks_v2/person_a_amazon_baseline.ipynb) reproduces the run locally.

## For B

Join each feature table to its platform’s user-fold file on `review_id`, validating a one-to-one match. Amazon IDs come from the source notebook export. Yelp IDs encode the zero-based row in its original feature parquet. File fingerprints are recorded in the model settings. In both exported tables, `label=1` is the positive class and `label=0` is the negative class. On Amazon, restrict analyses to `eligible=True` before using these labels. The legacy source assigns some unknown outcomes to its negative class. Eligibility follows the earlier Amazon audit and deduplication.

For the paired ablation use `eligible & ~product_holdout`, the feature order in the model settings and the same XGBoost settings as the baseline. For each outer test fold 0 through 4, use the next fold cyclically for choosing the F1-maximizing threshold and fit on the three remaining folds. For the frozen model, fit on folds 1 through 4 and select the threshold on fold 0. The GPU implementation checks the actual training device. The pooled confusion matrix sums predictions made with their respective validation cutoffs. In the metrics log, `pr_auc` is trapezoidal area and `ap` is average precision.

The [gain table](gain_importance.csv) includes every input, including ones receiving zero gain. `total_gain_share` measures each feature’s fraction of the reduction in training loss summed over tree splits. `gain_share` normalizes XGBoost’s average gain per split for each feature. The rating-family list is provisional until B reviews it. If its membership changes, sum the appropriate rows in the table.

## For C

For the Amazon product test, join on `review_id` and select `eligible & product_holdout`. Use the same product IDs as the existing audited Amazon run. For Yelp, use the complete Person A Yelp feature table for the all-review cohort. For the inflation cohort, keep every recommended review (`label=0`) plus filtered reviews (`label=1`) rated four or five stars. Preserve missing feature values when scoring. Both tables contain IDs and labels alongside the predictors, so select inputs using the exact feature list in the model settings.

```python
import json
from pathlib import Path

import pandas as pd
from xgboost import DMatrix, Booster

model_dir = Path("models/person_a")
spec = json.loads((model_dir / "model.json").read_text())
booster = Booster()
booster.load_model(model_dir / spec["weights"])
reviews = pd.read_parquet("data/features/yelp_person_a.parquet")
matrix = DMatrix(reviews[spec["features"]].astype("float32"))
scores = booster.predict(matrix)
flagged = scores >= spec["threshold"]
```

**Report holdout results by reviewer cohort.** 16.3% of the Amazon product-holdout rows (12,098) come from reviewers who also appear in development. Their positive-label rate is 71.3%, compared with 10.2% for the 62,094 other reviews. The frozen model's AP is 0.958 and 0.230, respectively. The gap persisted in a refit that excluded overlapping reviewers from training and threshold selection. The legacy features still include full-platform aggregates, which can incorporate evaluation-period activity. Report the all-review and new-reviewer figures together. [Cohort results](audit/cohort_metrics.csv) · [Reviewer-exclusion refit](audit/strict_retrain.csv)

**Include the rating-only reference.** The positive-label rate by star rating scores AP 0.629 on the product holdout, compared with 0.548 for Week 1. The paired interval spans −0.182 to +0.203. Fit the reference on the same Amazon training split, choose its cutoff on Amazon validation, and include it beside the full and ablated models. [Paired intervals](audit/paired_bootstrap_ap.csv)

**Use the Week 1 score for ranking.** In the top decile the mean score is 0.996 and the observed positive-label rate is 55.8%. The follow-up has lower Brier loss on this holdout (0.070 versus 0.112). Formal calibration remains a Week 3 task. [Score deciles](audit/calibration_deciles.csv) · [Brier scores](audit/brier.csv)

**Interpretation.** Reviewers are disjoint between the five user folds. Products are disjoint between the development pool and the Amazon product test. The source feature notebook used full-platform activity when calculating counts, bursts and several similarity measures. These retrospective features may include events after the review or from evaluation rows. Report this limitation with any score from these files. The positive label combines a five-star product/reviewer rule with Amazon removal.

## Follow-up Amazon experiment (exploratory)

This candidate was chosen on earlier development products and compared on a previously examined holdout. Against rating only, the AP gain was +0.193 (95% product-bootstrap interval +0.165 to +0.224) across the holdout and +0.209 (+0.176 to +0.240) among reviewers absent from development. The interval against the Week 1 model was +0.003 to +0.412. [Paired intervals](audit/paired_bootstrap_ap.csv)

The [executed comparison notebook](../../notebooks_v2/person_a_prospective.ipynb) tested six methods using earlier development reviews and three product-disjoint folds. XGBoost using 18 current-review features and seven earlier-product features led every selection fold. It uses the available reviews of the same product from strictly earlier calendar days. The review counts include earlier source records regardless of their eventual label eligibility.

On the previously examined product holdout, average precision rose from 0.548 to 0.822, precision from 69.8% to 88.5%, and recall fell from 69.1% to 66.6%. The paired 95% product-bootstrap interval for the average-precision difference was +0.003 to +0.412. For F1 the interval was −0.006 to +0.156. On the product’s first observed day, average precision was 0.780 for this model and 0.784 for Week 1 across 706 reviews. On reviews with a known Amazon removal outcome, precision rose from 31.0% to 38.9%. Each measurement used the existing Amazon validation threshold for its model. [Metrics](prospective_20261121T180956177896Z/metrics.csv) · [Paired intervals](prospective_20261121T180956177896Z/paired_intervals.csv) · [Removal check](prospective_20261121T180956177896Z/removal_check.csv)

**For B.** Rating-derived features account for 86.4% of the new tree's total gain. The new rating-derived history columns are `prior_mean_stars`, `prior_high_share`, `prior_low_share` and `abs_prior_star_gap`. Use the original W1-A-01 model and splits for the assigned paired ablation, and label any follow-up ablation separately. [Follow-up gain table](prospective_20261121T180956177896Z/tree_gain.csv)

**For C.** The notebook's `score_reviews(directory, source_frame)` function loads the [frozen candidate](prospective_20261121T180956177896Z/model.json). Supply a single platform’s original notebook feature rows with product ID, date, 1–5 rating, review text and its local text features. Include the available earlier reviews of the products being scored. Select test cohorts after generating scores so that filtering by the evaluation label cannot change the computed history. The chronological Amazon evaluation used reviews dated April 22, 2020 or later, with separate earlier training and threshold data. Moderation decision dates were unavailable, so the backtest assumes the earlier training labels could be used at the cutoff.

## Sensitivity to three held-out products

The primary comparison includes all 673 held-out products. Week 1 flagged all 4,162 reviews of `B01LSUQSB0`, `B076F5P28T` and `B07B7K7N3P`. Only 492 carried the positive combined label. The earlier-history model flagged 203. On the 670 remaining products, AP was 0.837 for Week 1, 0.841 for earlier history and 0.658 for rating only. On all products, the corresponding values were 0.548, 0.822 and 0.629. B identified the three products while studying holdout errors. Report their exclusion as an exploratory sensitivity check. Whole-platform counts include later activity and can behave differently on unusually large products.

## Draft for group review before sending to the professor

Professor, we are training our baseline on Amazon product reviews and checking it on held-out Amazon products. For the cross-platform evaluation, our search has yet to identify a publicly downloadable second retailer dataset with credible labels for individual English-language reviews. Yelp-NYC provides a platform-filtering label, although its reviews cover restaurants. May we use Yelp as our second platform and report both the full dataset and a comparison that keeps all recommended reviews and only four- or five-star filtered reviews? Our brief proposed highlighting the second comparison because the Amazon label includes a five-star rule. We will report both results and explain the different label definitions. If a second e-commerce retailer is required this semester, could you confirm that requirement so we can adjust the evaluation?
