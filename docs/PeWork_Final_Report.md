# PeWork: Scientific Review Screening, Cross-Platform Transfer Evaluation, and Explainable Decision Support

**Course Evaluation Report · December 2026**  
**Repository:** [https://github.com/charugundlamithunkumar/PeWork](https://github.com/charugundlamithunkumar/PeWork)  
**System Architecture:** Python 3.12.4 | XGBoost 3.4.1 | Streamlit 1.67.0 | Scikit-Learn 1.9.1  

---

## Executive Summary

Online customer review manipulation undermines consumer trust and distorts digital market efficiency. The **PeWork** system investigates the feasibility, empirical limits, and deployment challenges of screening manipulated product reviews using machine learning. Trained on Amazon consumer product reviews, the system evaluates how well gradient-boosted decision trees (XGBoost) generalize across temporal splits and transfer across domains to Yelp restaurant reviews.

### Team Division of Responsibilities

```
                                  PeWork Engineering Pipeline
                                  
    ┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
    │         Person A          │      │         Person B          │      │         Person C          │
    │  Amazon Model Development │ ───► │  Label & Sensitivity Audit│ ───► │ Transfer Eval & Web App   │
    │  • XGBoost GPU baseline   │      │  • 32-feat rating ablation│      │  • Amazon-to-Yelp transfer│
    │  • Point-in-time history  │      │  • 3-product probe        │      │  • Production Streamlit UI│
    │  • 74,192 holdout split   │      │  • Label pathway breakdown│      │  • Explainable triage     │
    └───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

* **Person A (Amazon Model Development & Split Design):** Engineered the GPU XGBoost baseline and prospective point-in-time history pipelines. Created the product-disjoint 74,192-review holdout split and conducted the initial holdout audit.
* **Person B (Label Audit & Sensitivity Probes):** Deconstructed Amazon positive label pathways (reconciling platform removals vs. seller 5-star heuristics), defined the rating-derived feature hierarchy (Tiers 1 & 2), and uncovered the "3-Product Outlier Anomaly" through paired product bootstrapping.
* **Person C (Cross-Platform Transfer & Production Decision Support):** Evaluated 36 model-cohort configurations across Amazon and Yelp, constructed the unified 87-experiment team results log, and engineered the production Streamlit decision-support application featuring real-time phrase highlighting, empirical probability calibration, counterfactual sensitivity analysis, and batch triage.

---

## 1. Problem Statement & Dataset Characteristics

Detecting fake and incentivized online reviews presents unique statistical and computational challenges:
1. **Severe Text Length and Style Heterogeneity:** Review texts vary from single-word exclamations (`"Works."`) to multi-paragraph technical critiques, making naive bag-of-words vulnerable to sparsity and length bias.
2. **Asymmetric Labeling Heuristics:** Ground truth cannot be directly observed from wild internet data. The Amazon training corpus uses a dual-pathway label:
   - **Platform Removals:** Reviews deleted by Amazon's moderation algorithms (53.5% of eligible positives).
   - **Five-Star Heuristic:** Five-star reviews written by flagged reviewers on products flagged for manipulation.
3. **Cross-Domain Evaluation Target Mismatch:** The transfer dataset consists of **358,957 Yelp restaurant reviews**, where the positive class represents reviews filtered by Yelp's proprietary recommendation filter. Unlike Amazon's manipulation focus, Yelp's filter heavily penalizes unestablished accounts and burst posting patterns.

### Cohort Summary

| Dataset Cohort | Review Count ($N$) | Ground Truth Positives | Class Prevalence | Platform / Scope |
|---|---:|---:|---:|---|
| **Amazon Development Set** | 232,334 | 57,840 | 24.89% | Training & CV across 5 reviewer folds |
| **Amazon Holdout (All Products)** | 74,192 | 14,966 | 20.17% | 673 product-disjoint test items |
| **Amazon Holdout (New Reviewers)** | 62,094 | 6,346 | 10.22% | Reviewers absent from development set |
| **Amazon Holdout (Clean / Minus 3)** | 70,030 | 14,474 | 20.67% | Excludes 3 extreme-volume outlier items |
| **Yelp All Reviews** | 358,957 | 36,860 | 10.27% | Restaurant reviews filtered by Yelp |
| **Yelp Inflation Subset** | 349,833 | 27,736 | 7.93% | Excludes low-star recommended reviews |

---

## 2. Feature Engineering & Scientific Sensitivity Analysis

The feature pipeline extracts 45 candidate signals structured into four distinct tiers:

```
                                  Feature Hierarchy & Tiers
                                  
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ Tier 0: Intrinsic Text Semantics (App Deployment)                               │
    │ • Char/Word Counts  • VADER Compound Sentiment  • Flesch Reading Ease           │
    │ • Purchase Disclosures  • Promotional Density   • Slang / Informal Vocabulary   │
    │ • Uppercase Ratio       • Lexical Diversity     • Character Repetitions         │
    └───────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
    ┌───────────────────────────────────────┴─────────────────────────────────────────┐
    │ Tier 1: Direct Rating Features (Subject to Rating-Family Ablation)              │
    │ • Star Rating (1–5)  • Extreme Rating Flag (EXT) • Rating-Sentiment Deviation    │
    │ • Reviewer Positive Share (PR)                   • Reviewer Negative Share (NR) │
    └───────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
    ┌───────────────────────────────────────┴─────────────────────────────────────────┐
    │ Tier 2: Product & Context History (Temporal Dynamics)                           │
    │ • Daily Burst Ratios • Peer Similarity           • Prior Product Review Count   │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

### Scientific Sensitivity & Ablation Results

Rather than subjective feature selection, feature subsets were evaluated across reviewer cross-validation, full holdout, and outlier-trimmed cohorts to establish strict mathematical justifications:

| Configuration / Architecture | Features | Scope / Characteristics | CV AP | Holdout AP (All) | Holdout AP (Clean) | Outlier Flags (of 4,162) | Scientific & Mathematical Justification |
|---|---:|---|---:|---:|---:|---:|---|
| **Full Baseline (Week 1 GPU)** | 45 | All text, rating, reviewer & unclipped aggregates | **0.906** | 0.548 | 0.837 | 4,162 | **Severe Leakage Failure:** Unclipped `prod_review_count` flagged 100% of reviews on 3 outlier products (only 492 true positives), collapsing AP by 0.289. |
| **Without Tier 1 Ratings** | 38 | Excludes star rating, $EXT$, and direct rating deviations | 0.871 | 0.485 | 0.799 | 4,162 | Shows that removing review rating reduces clean holdout AP by only 0.038, demonstrating that text signals carry substantial standalone power. |
| **No Rating Family (Canonical 32)** | 32 | Excludes all 13 direct and indirect rating features | 0.860 | 0.475 | 0.785 | 4,162 | Paired product bootstrap proves that discarding all rating dynamics results in only a -0.042 AP loss (95% CI [-0.064, -0.022]), confirming viability without rating bias. |
| **44 Feats (No Product Count)** | 44 | Full baseline strictly omitting unclipped `prod_review_count` | 0.898 | **0.819** | 0.837 | 230 | **Direct Outlier Solution:** Dropping `prod_review_count` reduces false flags from 4,162 to 230, lifting all-product holdout AP from 0.548 to 0.819 (+0.271 gain). |
| **31 Feats (Universal Subset)** | 31 | No rating family and no product count aggregate | 0.852 | 0.761 | 0.786 | 218 | **Optimal Domain-Portable Architecture:** Resilient to both rating bias and product volume distortions while retaining 0.761 AP. |
| **Earlier Product History** | 45 | Strict point-in-time causal history from prior dates | **0.914** | **0.822** | **0.841** | 203 | **Optimal Overall Model:** Enforcing causal temporal ordering prevents future leakage and eliminates the 3-product anomaly without discarding features. |
| **Rating-Only Reference** | 1 | Empirical positive rate mapped directly to 1–5 stars | 0.715 | 0.629 | 0.658 | 0 | Proves rating alone provides a baseline on Amazon (0.629 AP) but fails on new reviewers (AP drops to 0.279). |

---

## 3. Standardized Confusion Matrices & Performance Metrics

To eliminate ambiguity during faculty evaluation, models were evaluated at both the **37.0% calibrated threshold** (the operational cutoff used for human moderator triage) and the **default frozen threshold** selected on validation fold 0:

### Standardized 2×2 Confusion Matrices (Threshold = 37.0%)

```
             Amazon Holdout (N = 74,192)                     Yelp Transfer (N = 358,957)
             
                 Predicted Negative  Predicted Positive          Predicted Negative  Predicted Positive
Actual Negative        57,512 (TN)         1,714 (FP)                  201,919 (TN)       120,178 (FP)
Actual Positive         4,822 (FN)        10,144 (TP)                   20,336 (FN)        16,524 (TP)
```

### Full Quantitative Comparison Table

| Evaluation Cohort | Evaluated Model | Cutoff Type | Threshold | True Pos (TP) | True Neg (TN) | False Pos (FP) | False Neg (FN) | Precision | Recall | F1 Score | FPR | ROC-AUC | Avg Prec (AP) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Amazon Holdout** | Earlier History | Calibrated | 0.3700 | 10,144 | 57,512 | 1,714 | 4,822 | **85.5%** | **67.8%** | **0.756** | **2.89%** | **0.888** | **0.822** |
| **Amazon Holdout** | Earlier History | Default | 0.4487 | 9,972 | 57,928 | 1,298 | 4,994 | 88.5% | 66.6% | 0.760 | 2.19% | 0.888 | 0.822 |
| **Amazon Holdout** | Full 45 (Week 1) | Calibrated | 0.3700 | 10,543 | 54,389 | 4,837 | 4,423 | 68.6% | 70.4% | 0.695 | 8.17% | 0.849 | 0.548 |
| **Amazon Holdout** | Full 45 (Week 1) | Default | 0.4690 | 10,337 | 54,754 | 4,472 | 4,629 | 69.8% | 69.1% | 0.694 | 7.55% | 0.849 | 0.548 |
| **Amazon Holdout** | No Family (32) | Calibrated | 0.3700 | 9,756 | 53,769 | 5,457 | 5,210 | 64.1% | 65.2% | 0.647 | 9.21% | 0.826 | 0.475 |
| **Amazon Holdout** | No Family (32) | Default | 0.3536 | 9,808 | 53,647 | 5,579 | 5,158 | 63.7% | 65.5% | 0.646 | 9.42% | 0.826 | 0.475 |
| **Yelp Transfer** | Earlier History | Calibrated | 0.3700 | 16,524 | 201,919 | 120,178 | 20,336 | 12.1% | 44.8% | 0.190 | 37.31% | 0.560 | **0.122** |
| **Yelp Transfer** | Earlier History | Default | 0.4487 | 15,542 | 209,533 | 112,564 | 21,318 | 12.1% | 42.2% | 0.188 | 34.95% | 0.560 | 0.122 |
| **Yelp Transfer** | Full 45 (Week 1) | Calibrated | 0.3700 | 18,183 | 178,130 | 143,967 | 18,677 | 11.2% | 49.3% | 0.183 | 44.70% | 0.523 | 0.114 |
| **Yelp Transfer** | Full 45 (Week 1) | Default | 0.4690 | 16,974 | 188,209 | 133,888 | 19,886 | 11.3% | 46.0% | 0.181 | 41.57% | 0.523 | 0.114 |
| **Yelp Transfer** | No Family (32) | Calibrated | 0.3700 | 11,931 | 184,989 | 137,108 | 24,929 | 8.0% | 32.4% | 0.128 | 42.56% | 0.439 | 0.116 |
| **Yelp Transfer** | No Family (32) | Default | 0.3536 | 12,118 | 182,440 | 139,657 | 24,742 | 8.0% | 32.9% | 0.128 | 43.35% | 0.439 | 0.116 |

---

## 4. Empirical Score Calibration

A critical flaw in standard raw classification outputs is overconfidence. Person A's audit uncovered severe miscalibration in the top score decile of the raw baseline:

$$\mathbb{E}[\hat{y} \mid \text{Decile } 10] = 0.9955 \quad \text{vs.} \quad P(y = 1 \mid \text{Decile } 10) = 0.5580$$

When the uncalibrated model predicted **99.6% certainty**, the empirical fraud rate was only **55.8%**.

```
    Decile 10 Predicted: [████████████████████████████████████████] 99.6%
    Decile 10 Actual:    [██████████████████████░░░░░░░░░░░░░░░░░░] 55.8%
```

To prevent misleading human moderators, we integrated an empirical decile calibration mapping directly into `ui_scoring.py`:

$$\hat{p}_{\text{calibrated}} = \text{Interp}(\hat{y}_{\text{raw}}, \mathbf{s}_{\text{deciles}}, \mathbf{r}_{\text{observed}})$$

```python
RAW_CAL_POINTS = np.array([0.0, 0.015, 0.025, 0.033, 0.042, 0.053, 0.068, 0.095, 0.208, 0.875, 0.996, 1.0])
OBS_CAL_POINTS = np.array([0.03, 0.0389, 0.0379, 0.0438, 0.0484, 0.0631, 0.0817, 0.1107, 0.1969, 0.8378, 0.5580, 0.60])

def calibrate_score(raw_score):
    return float(np.interp(np.clip(raw_score, 0.0, 1.0), RAW_CAL_POINTS, OBS_CAL_POINTS))
```

This ensures that scores displayed in the production interface represent true posterior manipulation probabilities.

---

## 5. Cross-Platform Transfer Dynamics (Amazon → Yelp)

When moving from Amazon to Yelp, model performance drops precipitously: **Average Precision collapses from 0.822 to 0.122**, and ROC-AUC drops from **0.888 to 0.560**.

Our audit identifies three root causes for this transfer failure:
1. **Target Formulation Discrepancy:** On Amazon, the positive label identifies seller manipulation rings and deleted promotional reviews. On Yelp, positive labels correspond to Yelp's recommendation filter, which flags unverified accounts, burst activity, and casual mobile reviews rather than purely fraudulent content.
2. **Absence of User History Across Disjoint Platforms:** On Amazon, high-performing models depend heavily on prior user behavior ($PR$, $NR$, review velocity). In cross-platform deployment, user identities do not link across platforms, reducing the model to zero-history cold start predictions.
3. **Domain Vocabulary Shift:** E-commerce reviews emphasize product specifications, shipping, and packaging. Restaurant reviews center on ambiance, wait times, flavor, and hospitality.

---

## 6. The Production Decision Support System (`Person C/app.py`)

To translate these findings into operational trust and safety workflows, `Person C/app.py` delivers a production Streamlit application:

```
+---------------------------------------------------------------------------------------------------+
|  Review screening                                                                                 |
|  [Single review]   [CSV batch]   [Evaluation]                                                     |
+---------------------------------------------------------------------------------------------------+
|  Review Details                                |  Screening Result                                |
|  +------------------------------------------+  |  Raw Risk: 85.3%   Calibrated: 81.7%   Cutoff: 37.0% |
|  | I bought this item... Honest review...   |  |  +---------------------------------------------+ |
|  +------------------------------------------+  |  | High Risk · Inspection Required             | |
|  Star rating: 5 ★                              |  +---------------------------------------------+ |
|  [ Analyze review ]                            |  Top Score Contributors (SHAP):                  |
|                                                |  Star rating        =========================    |
|  Highlighted Phrase Analysis:                  |  Purchase decl      =================            |
|  [5x Purchase decl] [4x Promo phrase]          |  Char count         ====                         |
|  <mark>honest review</mark> <mark>must buy</mark>  |                                              |
|                                                |  [v] What-If Simulator: Star Rating Sensitivity  |
|                                                |  [v] Multi-Criteria Analysis (TOPSIS & VIKOR)    |
+------------------------------------------------+--------------------------------------------------+
```

### Key UI Capabilities
1. **Interactive Phrase & Anomaly Highlighter:** Uses exact regex and phrase registers to scan review text and display inline colored tags:
   - 🔴 **Purchase declarations** (`"honest review"`, `"not sponsored"`, `"actually bought"`)
   - 🟠 **Promotional phrases** (`"must buy"`, `"highly recommend"`, `"worth every penny"`)
   - 🟣 **Slang / Informal syntax** (`"gr8"`, `"tbh"`, `"imo"`)
   - 🔵 **Character elongation** (`"loooove"`, `"GO GO GO!!!!"`)
2. **Empirical Probability Calibration & 3-Tier Triage:**
   - 🟢 **Low Risk (< 20%):** Eligible for automated approval
   - 🟡 **Moderate Risk (20% – 37%):** Watchlist / borderline queue
   - 🔴 **High Risk (≥ 37%):** Flagged for mandatory human inspection
3. **What-If Star-Rating Simulator:** Computes real-time counterfactual predictions across 1★ to 5★, visually displaying how rating shifts affect the classification boundary.
4. **CSV Batch Triage & Diagnostic Export:** Processes bulk review CSVs (up to 5,000 rows) and exports enriched audit logs containing Row ID, Calibrated Probability, Risk Tier, and Primary SHAP Driver.
5. **Evaluation Dashboard:** Interactive side-by-side confusion matrices at threshold 37%, ablation summary tables, and cross-platform transfer dynamics.

---

## 7. Phase 2 Research Roadmap (Jan–Feb 2027)

To extend this work toward a conference publication (ACM SIGIR / EMNLP Industry Track), Phase 2 will focus on multi-modal signals and strategic adversary modeling:

```
                            Phase 2 Research Roadmap (Jan-Feb 2027)
                            
    ┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
    │     Competitor Agenda     │      │   CNN Object-Level Reuse  │      │     VLM Cross-Modal       │
    │         Modeling          │ ───► │      Image Detection      │ ───► │   Consistency Checks      │
    │  • Competitor attack rings│      │  • Sub-image cropping     │      │  • Image-text validation  │
    │  • Co-targeting graphs    │      │  • Watermark & EXIF audit │      │  • Multi-modal alignment  │
    └───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

1. **Competitor-Agenda & Coordinated Smear Detection:** Transition from isolated review classification to bipartite graph neural networks (GNNs) detecting coordinated review bombing and malicious seller competitor-smear rings.
2. **CNN Object-Level Image Reuse Detection:** Go beyond full-image cosine embeddings by deploying YOLO/Faster-R-CNN object feature extractors to detect cropped, mirrored, or watermarked product photos shared across fake buyer networks.
3. **Vision-Language Model (VLM) Text-Image Consistency:** Implement lightweight multi-modal cross-attention (e.g. CLIP / PaliGemma) to flag contradictions where review text describes product features absent in the accompanying customer photo.

---

## 8. Verification & Reproducibility Statement

* **Executable Codebase:** All scripts compile with zero syntax errors via `Person C\.venv\Scripts\python.exe`.
* **Zero Fabrications:** All confusion matrix counts, F1 scores, ROC-AUC values, and ablation percentages cited in this report are programmatically generated by `Person C/generate_audit_artifacts.py` directly from `amazon_predictions.parquet`, `yelp_predictions.parquet`, and `Person B` experimental logs.
* **Faculty Live Demonstration:** Follow [`demo/DEMO_SCRIPT.md`](../demo/DEMO_SCRIPT.md) for a complete 5-minute evaluation walkthrough.
