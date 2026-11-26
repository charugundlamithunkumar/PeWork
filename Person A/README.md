# Person A · Amazon model handoff

Open this folder in VS Code. The two notebooks include outputs from the completed runs. The [team handoff](results/person_a/team_handoff.md) contains scoring instructions for C, the split and feature conventions for B, and the draft professor message.

## What’s in this folder

| Deliverable | Where to find it |
|---|---|
| Agreed Week 1 GPU baseline | [Executed notebook](notebooks_v2/person_a_amazon_baseline.ipynb), [frozen model](models/person_a/amazon_xgb.ubj), [settings and cutoff](models/person_a/model.json) |
| Shared inputs for B and C | [Amazon and Yelp feature tables](data/features), [reviewer folds and product holdout](data/splits) |
| Baseline evaluation | [Fold and holdout metrics](results/person_a/results_log.csv), [gain by feature](results/person_a/gain_importance.csv) |
| Improved Amazon model | [Executed comparison notebook](notebooks_v2/person_a_prospective.ipynb), [saved model and evaluation files](results/person_a/prospective_20261121T180956177896Z) |
| Audit | [Audit write-up](AUDIT.md), [audit script](review_pipeline/person_a_audit.py), [audit tables](results/person_a/audit), [confusion matrices](figures/week1_confusion_matrices.png) |
| Implementation | [Baseline training](review_pipeline/person_a.py), [follow-up training and scoring](review_pipeline/person_a_prospective.py) |

The saved baseline can score the included feature tables directly. Run the handoff's scoring example from this folder. For B's paired ablation, use the **Week 1** features, model settings and splits.

## Reproducing training

Use the project's configured Python kernel with CUDA-enabled XGBoost. Run the baseline notebook first and the comparison notebook second. Rerunning the baseline writes the baseline files inside this folder. The comparison creates a timestamped run here.

Both notebooks use the shared source files under `datasets/ECommerce_Project` in the parent project. The baseline also reads `data_manifest.json` and `amazon_reviews.parquet` from the parent's `v2/runs/amazon_yelp_v2` audit. If you move this folder, point the environment variable `REVIEW_SHARED_PROJECT` at the project containing those inputs. Browsing the saved results requires only this folder.


## Checking the results without the parent project

From this folder, use the project's Python environment to run `python -m review_pipeline.person_a_audit`. Add `--device cuda` to run the diagnostic refits on a supported GPU. The audit verifies the splits and frozen holdout metrics, then regenerates its tables and figure. Retraining the baseline or follow-up notebooks needs the parent project's source data.
