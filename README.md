# PeWork · Review screening

PeWork trains review-screening models on Amazon product reviews and evaluates how they transfer to Yelp. The repository contains three team folders with the saved models, experiment results, and code. Amazon positives use the project’s combined label. Yelp positives are reviews filtered by Yelp.

## What’s in each folder

| Folder | Work | Start here |
|---|---|---|
| **Person A** | Amazon model development. Includes the executed GPU XGBoost baseline and earlier-product-history notebooks, shared Amazon and Yelp feature tables, reviewer folds, Amazon product holdout, frozen weights, feature importances, and the model audit. | [Person A README](Person%20A/README.md) · [Model audit](Person%20A/AUDIT.md) |
| **Person B** | Amazon label audit and rating-feature ablations. Includes counts for the different label pathways, the list of rating-derived features, CPU refits with and without those features, the frozen 32-feature model, and sensitivity analyses. | [Person B README](Person%20B/README.md) · [Label audit and ablation report](Person%20B/PERSON_B_REPORT.md) |
| **Person C** | Cross-platform evaluation and the Streamlit demo. Includes four Amazon holdout cohorts, two Yelp cohorts, model scores by review, transfer metrics, the shared results log, and the UI code and weights. | [Person C README](Person%20C/README.md) · [Transfer results](Person%20C/transfer_results.csv) |

Person A supplies the saved models and evaluation splits. Person B uses those splits for the rating-feature ablations. Person C evaluates the frozen models on Amazon and Yelp and checks that the 32-feature model reproduces Person B’s Amazon holdout scores.

## Run the demo or inspect the results

- **Try the Streamlit UI.** Use the setup commands for VS Code and PowerShell. The included model and evaluation files support single-review scoring, CSV batch scoring, and browsing the Amazon and Yelp results. [UI setup](Person%20C/README.md#review-screening-ui)
- **Open the experiment results.** The shared CSV contains fold metrics, holdout and transfer results, thresholds, and confusion counts. The transfer table contains 36 model and cohort evaluations. [Shared results log](Person%20C/results_log_team_with_C.csv) · [Transfer results](Person%20C/transfer_results.csv)
- **Rerun an experiment.** Follow the corresponding folder README for dependencies and input paths. The UI and saved reports use files in this repository. Retraining, rerunning the raw label audit, and rescoring the full source timelines require the source inputs named in the relevant instructions.
