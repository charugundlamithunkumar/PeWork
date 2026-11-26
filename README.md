# PeWork · Cross-Platform Review Screening & Fraud Detection

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 📄 **Academic Report:** [`docs/PeWork_Final_Report.md`](docs/PeWork_Final_Report.md) ([Printable HTML](docs/PeWork_Final_Report.html))  
> 🎬 **Faculty Demo Script:** [`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md) ([Sample CSV](demo/sample_reviews.csv))  
> 📊 **Audit Artifacts:** [`Person C/audit_artifacts/`](Person%20C/audit_artifacts/)

**PeWork** is a machine learning pipeline and interactive web application designed to screen and detect fraudulent, manipulated, or incentivized product reviews. The system trains gradient-boosted decision trees (XGBoost) on Amazon e-commerce product reviews and evaluates their generalization and transferability across platforms to Yelp restaurant reviews.

---

## 📌 Project Overview & Team Structure

The repository is modularized into three core team components:

```
PeWork/
├── Person A/   # Amazon model development, baseline training, product holdouts & audits
├── Person B/   # Label pathway audit, rating-feature ablations & sensitivity probes
└── Person C/   # Amazon-to-Yelp transfer evaluation, shared log & interactive Streamlit UI
```

| Component | Focus Area | Deliverables & Key Assets |
|---|---|---|
| **Person A** | **Amazon Model Development** | • Executed GPU XGBoost baseline and temporal product-history notebooks<br>• Shared Amazon and Yelp feature tables & reviewer cross-validation folds<br>• Product holdout splits, SHAP feature importance, and diagnostic audit reports |
| **Person B** | **Label Audit & Feature Ablations** | • Audit of Amazon positive label pathways (removals vs. 5-star manipulation heuristics)<br>• Definition of rating-derived feature families (Tiers 1 & 2)<br>• 32-feature CPU refits, paired product bootstrapping, and 3-product sensitivity probes |
| **Person C** | **Transfer Evaluation & Web Dashboard** | • 36 evaluations across 4 Amazon holdout cohorts and 2 Yelp cohorts<br>• Cross-platform transfer analysis and shared team experiment log (`results_log_team_with_C.csv`)<br>• Production-ready interactive Streamlit screening application (`app.py`) |

---

## 🚀 Key Features in the Interactive Web Dashboard (`Person C`)

The project includes an interactive web application located in `Person C/` designed for human moderators and trust & safety teams:

### 1. 🔍 Single Review Real-Time Screening
* **Live Risk Scoring:** Predicts an uncalibrated Amazon-label manipulation risk score and compares it against the frozen validation threshold (`36.97%`).
* **Calibrated Empirical Probability:** Maps raw model scores through holdout decile calibration curves to reflect true real-world likelihood of manipulation.
* **Risk Triage Tiers:**
  * 🟢 **Low Risk (< 20%):** Eligible for automated approval.
  * 🟡 **Moderate Risk (20% – 37%):** Watchlist / borderline.
  * 🔴 **High Risk (≥ 37%):** Flagged for manual human inspection.
* **SHAP Score Contributors:** Renders an Altair bar chart highlighting the top 3 signed feature contributions moving the prediction in raw log-odds.

### 2. 🎨 Interactive Pattern & Keyword Highlighter
* Analyzes raw review text in real-time and visualizes detected signals with colored badges:
  * 🔴 **Purchase Declarations:** (e.g., *"honest review"*, *"not sponsored"*, *"actually bought"*, *"genuine review"*).
  * 🟠 **Promotional Keywords:** (e.g., *"must buy"*, *"highly recommend"*, *"worth every penny"*, *"life changing"*).
  * 🟣 **Slang / Informal Language:** (e.g., *"gr8"*, *"tbh"*, *"imo"*, *"omg"*).
  * 🔵 **Character Repetition / Elongation:** (e.g., *"GO GO GO!!!!"*, *"loooove"*).

### 3. ⚡ "What-If" Counterfactual Simulator
* Dynamically simulates how the review would score if the reviewer had given a different star rating (**1 ★ to 5 ★**).
* Displays a side-by-side sensitivity bar chart with the decision cutoff line, showing how rating changes impact model confidence.

### 4. 📊 Multi-Criteria Decision Making (TOPSIS & VIKOR)
* Includes separate MCDM ranking assessments computed using frozen Amazon development reference bounds, providing distance-to-ideal and weighted regret metrics.

### 5. 📁 CSV Batch Triage & Enriched Reporting
* Upload CSV files containing `review_text` and `rating` (up to 5,000 rows).
* Computes batch statistics across 5 KPI cards: **Scored**, **High Risk 🚩**, **Moderate ⚠️**, **Low Risk ✅**, and **Invalid rows**.
* Generates an enriched downloadable CSV with **Calibrated Risk**, **Risk Tier**, and the **Primary Diagnostic Factor** for each row.

### 6. 🔬 Evaluation & Cross-Platform Transfer Explorer
* Interactive dashboard comparing 6 model variants across 6 distinct Amazon and Yelp cohorts.
* Interactive Average Precision (AP) comparative bar charts, ROC-AUC, Precision, Recall, F1 scores, and 2×2 confusion matrices.
* Deep-dive documentation on the root causes behind cross-platform performance shifts.

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
* 64-bit Python **3.12** or higher
* Git

### Installation & Launch

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/charugundlamithunkumar/PeWork.git
   cd PeWork
   ```

2. **Navigate to `Person C` and set up the virtual environment:**
   ```powershell
   cd "Person C"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled in PowerShell, first run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

3. **Install dependencies:**
   ```powershell
   pip install -r requirements-ui.txt
   ```

4. **Launch the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```
   Open **`http://localhost:8501`** in your browser.

---

## 📈 Summary of Transfer Evaluation Findings

The models were evaluated across six frozen test cohorts:

| Test Cohort | Review Count | Earlier History AP | Week 1 (45 feats) AP | 32-Feat (No Rating Fam) AP |
|---|---:|---:|---:|---:|
| **Amazon · All Held-out Products** | 74,192 | **0.822** | 0.548 | 0.475 |
| **Amazon · New Reviewers Only** | 62,094 | **0.489** | 0.230 | 0.193 |
| **Amazon · 3 Products Excluded** | 70,030 | **0.841** | 0.837 | 0.785 |
| **Amazon · 3 Excluded, New Reviewers** | 58,307 | **0.518** | 0.487 | 0.339 |
| **Yelp · All Restaurant Reviews** | 358,957 | **0.122** | 0.114 | 0.116 |
| **Yelp · Inflation Subset** | 349,833 | **0.118** | 0.095 | 0.086 |

### Core Takeaways:
1. **The 3-Product Anomaly:** Person B discovered that 3 high-volume products on Amazon accounted for all 4,162 reviews being flagged by Week 1 models due to unclipped aggregate counts (`prod_review_count`). Excluding them restores baseline holdout AP from **0.548** to **0.837**.
2. **Cross-Platform Transfer Collapse:** Transferring Amazon-trained models to Yelp causes AP to drop to **~0.12**. This is driven by target definition discrepancies (Amazon combined manipulation heuristic vs. Yelp proprietary recommendation filter) and the absence of cross-platform historical timeline linkages.

---

## 📂 Repository File Index

* [`Person A/`](Person%20A/README.md)
  * `notebooks_v2/`: Executed baseline and temporal history Jupyter notebooks.
  * `review_pipeline/`: Feature engineering and model training scripts.
  * `AUDIT.md`: In-depth audit report on holdout leakage and reviewer splits.
* [`Person B/`](Person%20B/README.md)
  * `PERSON_B_REPORT.md`: Comprehensive report on label pathways and rating family ablations.
  * `results/person_b/models/`: Saved 32-feature model bundles and manifests.
* [`Person C/`](Person%20C/README.md)
  * `app.py`: Streamlit web interface with real-time analysis, text highlighter, and simulators.
  * `ui_scoring.py`: Scoring utilities, SHAP explanations, calibration, and batch processor.
  * `ui_assets/`: Pre-bundled model weights, manifests, and MCDM ranking references.
  * `transfer_results.csv`: Complete metrics across 36 model/cohort evaluations.
  * `audit_artifacts/`: Standardized confusion matrices and feature sensitivity tables.
* [`docs/`](docs/PeWork_Final_Report.md)
  * [`PeWork_Final_Report.md`](docs/PeWork_Final_Report.md) · [Print-ready HTML](docs/PeWork_Final_Report.html): Complete formal academic evaluation report.
* [`demo/`](demo/DEMO_SCRIPT.md)
  * [`DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md): 5-minute structured faculty walkthrough script.
  * [`sample_reviews.csv`](demo/sample_reviews.csv): 10 tested benchmark reviews hitting all three risk tiers.

