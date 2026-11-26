# Person B handoff (label audit, rating family, ablation)

Start with [PERSON_B_REPORT.md](PERSON_B_REPORT.md). It is one document with the findings, the corrections to earlier claims, and next steps.

## What’s in this folder

| What | Where |
|---|---|
| Rating-derived feature list (tiers and reasons) | [rating_family.json](results/person_b/rating_family.json), [rating_family.py](review_pipeline/rating_family.py) |
| Paired ablation (CV, holdout, bootstrap) | [person_b_ablation.py](review_pipeline/person_b_ablation.py), `results/person_b/ablation_*.csv` |
| Same ablation without the 3 dominating products | [person_b_trimmed_ablation.py](review_pipeline/person_b_trimmed_ablation.py), `ablation_holdout_trimmed.csv` |
| Feature groups, size-shift test | [person_b_feature_groups.py](review_pipeline/person_b_feature_groups.py), [person_b_size_shift.py](review_pipeline/person_b_size_shift.py), `groups_*.csv`, `size_shift_test.csv` |
| Three-product problem | [person_b_holdout_influence.py](review_pipeline/person_b_holdout_influence.py), [person_b_product_context_probe.py](review_pipeline/person_b_product_context_probe.py), [person_b_range_test.py](review_pipeline/person_b_range_test.py) |
| Label audit | [Audit code](review_pipeline/person_b_label_audit.py), [label pathway counts](results/person_b/label_audit_paths.csv), [raw value counts](results/person_b/label_audit_raw_value_counts.csv), [cross-tabulations](results/person_b/label_audit_raw_crosstabs.csv) |
| Frozen CPU models | [Model bundles](results/person_b/models), [32-feature manifest](results/person_b/models/minus_family_32.json), [saved holdout scores](results/person_b/minus_family_32_holdout.parquet) |
| Verified copy of the canonical A/B/C results log | [results_log_team.csv](results/results_log_team.csv) |

## Reproducing the handoff

From the project root, run the following in the same Python environment as C. The default package lookup supports both the checked-in layout and an extracted B package next to Person A. To use a different A folder, add `--package <path>`.

```powershell
python -m "Person B.review_pipeline.person_b_label_audit" --raw datasets/ECommerce_Project/amazon_data.csv
python -m "Person B.review_pipeline.person_b_ablation" --device cpu
python -m "Person B.review_pipeline.person_b_trimmed_ablation"
python -m "Person B.review_pipeline.person_b_feature_groups"
python -m "Person B.review_pipeline.person_b_holdout_influence"
python -m "Person B.review_pipeline.person_b_product_context_probe"
python -m "Person B.review_pipeline.person_b_range_test"
python -m "Person B.review_pipeline.person_b_size_shift"
python -m "Person C.run_transfer"
python -m "Person C.build_team_log"
```

The ablation command recreates the four frozen CPU models, reviewer-fold metrics, holdout scores and paired product bootstrap. C reads the [32-feature manifest](results/person_b/models/minus_family_32.json) and checks its predictions against [B's saved holdout scores](results/person_b/minus_family_32_holdout.parquet). The other B experiments are sensitivity analyses. The raw audit needs the original Amazon CSV. C's history scoring requires the two original feature parquets. All saved evaluation tables and prediction files can be read directly.
