# Person C | Amazon-to-Yelp transfer evaluation

[transfer_results.csv](transfer_results.csv) contains 36 evaluations across six cohorts. Each row identifies the model, validation threshold, weights checksum, cohort membership file, prevalence, ranking metrics and confusion counts. The positive class is Amazon's combined label or Yelp's filtered status, depending on the test platform. The earlier-history model's AP is 0.822 on all held-out Amazon products and 0.122 on all Yelp restaurants.

## What’s in this folder

| Files | Contents |
|---|---|
| [test_sets/](test_sets) · [test_set_summary.csv](test_set_summary.csv) | Review IDs and labels for four Amazon holdout cohorts and two Yelp cohorts, plus their sizes and positive rates. |
| [transfer_results.csv](transfer_results.csv) · [results_log_team_with_C.csv](results_log_team_with_C.csv) | Transfer metrics and the combined experiment log for the team. |
| [amazon_predictions.parquet](amazon_predictions.parquet) · [yelp_predictions.parquet](yelp_predictions.parquet) | Per-review scores for reproducing the transfer metrics. |
| [feature_availability.csv](feature_availability.csv) · [models/](models) | Feature coverage across platforms, the two product-count ablations, and the rating-only reference. |
| [run_transfer.py](run_transfer.py) · [build_team_log.py](build_team_log.py) | Scripts to score the six cohorts and rebuild the shared log. |
| [app.py](app.py) · [ui_scoring.py](ui_scoring.py) · [ui_assets/](ui_assets) · [requirements-ui.txt](requirements-ui.txt) | Streamlit UI, scoring code, frozen text-and-rating model, TOPSIS/VIKOR references, and app dependencies. |

## Review screening UI

The [Streamlit app](app.py) runs locally with the included model weights and evaluation results. You need Git and a 64-bit Python installation. The setup below uses PowerShell in VS Code and was verified with Python 3.14.

If you already have a checkout, open it in VS Code and run `git pull`. For a fresh checkout:

```powershell
git clone https://github.com/charugundlamithunkumar/PeWork.git
Set-Location PeWork
```

From the repository root, create a virtual environment for the app and launch it:

```powershell
Set-Location "Person C"
python -m venv .venv
.\.venv\Scripts\python.exe -E -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -E -m streamlit run app.py --server.address=127.0.0.1
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) or the URL printed in the terminal. Try **Single review → Use example → Analyze review**. Keep the terminal open while using the app. Press **Ctrl+C** to stop it. On later runs, open a terminal in `Person C` and use the launch command.

- **Single review** accepts text and a 1–5 star rating. It displays an Amazon-label risk score, the frozen threshold, a human-review flag, and up to three signed XGBoost contributions. Expand TOPSIS and VIKOR for scores, thresholds and criterion details. The example picker includes ordinary, ambiguous and five-star neutral reviews.
- **CSV batch** accepts a UTF-8 CSV with `review_text` and `rating` columns, up to 5,000 reviews. Use the downloadable template. The results identify rows by position and report validation errors. The scored export contains the rating, score, decision and any input issue.
- **Evaluation** shows the six frozen transfer models and references across Amazon and Yelp, with average precision, ROC-AUC, trapezoidal PR-AUC, precision, recall, F1 and confusion counts.

The UI ships the v2 model and the frozen Amazon TOPSIS/VIKOR references in [ui_assets](ui_assets). Live scoring uses features available from review text and stars. Its score is uncalibrated and its validation cutoff is 0.36965659260749817. The team’s earlier-history model requires chronologically earlier reviews for the same product and appears in the evaluation view. You can use all three UI views with the included artifacts and your own review inputs.

## Reproduce the scores locally

Open the repository root containing `Person A`, `Person B` and `Person C` in VS Code. The verified CPU scoring run used XGBoost 3.4.1, scikit-learn 1.9.0, pandas 3.1.1 and NumPy 2.5.5. Keep the original `amazon_features.parquet` and `yelp_features.parquet` under `datasets/ECommerce_Project`, or pass their directory using `--dataset`. The script verifies both files against A's source hashes. You can rebuild the saved results log using the files in this repository.

```powershell
python -m pip install numpy pandas scipy scikit-learn xgboost pyarrow joblib matplotlib vaderSentiment
python -m "Person C.run_transfer"
python -m "Person C.build_team_log"
```

The transfer command loads the shipped models, scores both platforms and checks the Amazon results against the owners' predictions. The log command rebuilds the shared log inside B's and C's folders. To reproduce C's training step, add `--train-variants` to the transfer command. This refits the two product-count ablations and rating reference on the Amazon development folds and replaces their local artifacts.

## Recompute a metric from the saved scores

[Amazon predictions](amazon_predictions.parquet) and [Yelp predictions](yelp_predictions.parquet) each have one row per review and one score column per model. Join the prediction file to a [cohort file](test_sets) by `review_id` and `label`. A positive flag is a score greater than or equal to that model's threshold. The inflation subset contains every recommended Yelp review and filtered reviews with four or five stars. History is always computed before that filter.

```python
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

scores = pd.read_parquet("Person C/yelp_predictions.parquet")
keys = pd.read_parquet("Person C/test_sets/yelp_inflation.parquet")
rows = keys.merge(scores, on=["review_id", "label"], validate="one_to_one")
assert len(rows) == len(keys)
print(average_precision_score(rows.label, rows.earlier_history))
print(confusion_matrix(rows.label, rows.earlier_history >= 0.4487259089946747, labels=[0, 1]))
```

The 45-feature baseline and the earlier-history model belong to A. The 32-feature rating-family ablation belongs to B. C's 44- and 31-feature CPU variants omit `prod_review_count` and have their own saved weights. The rating-only reference maps each star level to its positive rate in Amazon training folds 1–4. These six scoring rules share the existing Amazon validation protocol. The rebuilt log records historical experiments separately. [Model artifacts for C's variants](models) · [Feature availability](feature_availability.csv) · [Shared results log](results_log_team_with_C.csv)
