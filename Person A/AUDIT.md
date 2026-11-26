# Person A audit · November 23, 2026

The audit reloaded the frozen Week 1 model, verified the Amazon and Yelp splits, and compared Week 1, a rating-only reference, and the follow-up model on the same 74,192 held-out Amazon reviews. It also reran the Week 1 fit after excluding every holdout reviewer from both development and threshold selection. [Audit script](review_pipeline/person_a_audit.py) · [Generated tables](results/person_a/audit)

| Model | All products AP | Reviewers absent from development AP | Reviewers present in development AP |
|---|---:|---:|---:|
| Week 1 XGBoost | 0.548 | 0.230 | 0.958 |
| Rating only | 0.629 | 0.279 | 0.849 |
| Follow-up using earlier product history | 0.822 | 0.489 | 0.981 |

The holdout is product-disjoint. Reviewers who also appear in development account for 12,098 holdout rows (16.3%). Their positive label rate is 71.3%, compared with 10.2% among the other 62,094 rows. Average precision is sensitive to this difference in prevalence. Within the product holdout, both cohorts are relevant and answer different deployment questions. The audit also reproduces the Week 1 confusion counts, ROC-AUC of 0.8493, and trapezoidal PR-AUC of 0.5477. [Split checks](results/person_a/audit/integrity_checks.csv) · [Cohort metrics](results/person_a/audit/cohort_metrics.csv) · [Confusion matrices](figures/week1_confusion_matrices.png)

The paired 95% product-bootstrap interval for the rating-only AP advantage over Week 1 was −0.182 to +0.203. The follow-up gained 0.193 AP over rating only on the full holdout (95% interval +0.165 to +0.224) and 0.209 among reviewers absent from development (+0.176 to +0.240). Model selection used development products, and the follow-up comparison reused an already examined holdout. [Paired intervals](results/person_a/audit/paired_bootstrap_ap.csv)

The GPU refit that excluded overlapping reviewers reduced full-holdout AP from 0.548 to 0.530. The gap between reviewer cohorts persisted (0.953 versus 0.224). This shows the gap can arise even when those reviewers are excluded from supervised fitting and threshold selection. The refit reuses Week 1’s full-platform aggregates, so it cannot rule out evaluation rows or later activity entering the predictors. The follow-up constructs its product history from strictly earlier calendar days. [Refit metrics](results/person_a/audit/strict_retrain.csv)

At the top Week 1 score decile, mean score was 0.996 and the observed positive rate was 55.8%. The Brier scores for Week 1 and the follow-up were 0.112 and 0.070 on this holdout. Treat Week 1 scores as ranking signals. These diagnostics describe the combined Amazon label, and the deciles split tied scores in input order. [Score deciles](results/person_a/audit/calibration_deciles.csv) · [Brier scores](results/person_a/audit/brier.csv)

Among reviews with recorded Amazon removal status, the follow-up reached 38.9% precision and 0.337 AP at its existing cutoff, with 11.7% removals in the cohort. Recorded removal and the combined training label are different proxies for manipulation. [Removal check](results/person_a/prospective_20261121T180956177896Z/removal_check.csv)

The audit script regenerates the tables and confusion figure using the included package files. Trapezoidal PR-AUC and average precision are reported separately. The main Week 1 results log continues to document the original run, and audit rows are available in its schema for team reporting. [Audit log rows](results/person_a/audit/results_log_audit_rows.csv)

## Sensitivity to three held-out products

The primary comparison includes all 673 held-out products. Week 1 flagged all 4,162 reviews of `B01LSUQSB0`, `B076F5P28T` and `B07B7K7N3P`. Only 492 carried the positive combined label. The earlier-history model flagged 203. On the 670 remaining products, AP was 0.837 for Week 1, 0.841 for earlier history and 0.658 for rating only. On all products, the corresponding values were 0.548, 0.822 and 0.629. B identified the three products while studying holdout errors. Report their exclusion as an exploratory sensitivity check. Whole-platform counts include later activity and can behave differently on unusually large products.
