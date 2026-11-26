# PeWork · Faculty Live Evaluation & Demo Script (5 Minutes)

This script provides a structured, time-stamped walkthrough for demonstrating the **PeWork** review screening system during the December 2026 faculty evaluation.

---

## ⏱️ Timeline Overview

| Time | Section | Purpose |
|---|---|---|
| **0:00 – 0:45** | System Launch & Architecture Overview | Introduce the team scope (Person A, B, C) and launch the app |
| **0:45 – 2:15** | Single Review Analysis & Explainability | Demo 3 reviews (Genuine, Borderline 5★, Manipulated) + Highlighter + SHAP |
| **2:15 – 3:00** | What-If Star-Rating Simulator | Demonstrate rating sensitivity and counterfactual reasoning |
| **3:00 – 3:45** | CSV Batch Triage & Diagnostic Audit | Upload `demo/sample_reviews.csv`, show 5 KPI cards, download audit log |
| **3:45 – 5:00** | Evaluation Dashboard & Scientific Findings | Show 37% Confusion Matrices, Feature Sensitivity Audit & Transfer Drop |

---

## 🚀 Step 1: Launch the Application (0:00 – 0:45)

1. Open PowerShell in the project directory:
   ```powershell
   cd "C:\Users\Varshith M Gowda\Downloads\PeWork-main\Person C"
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
2. Navigate to **`http://localhost:8501`** in the browser.
3. **Presenter Talking Point:**
   > *"Good morning committee. PeWork is a production-grade machine learning system designed to screen fraudulent, incentivized, and manipulated reviews. Our codebase is structured across three core modules: Person A developed our GPU XGBoost baselines and temporal history splits; Person B conducted our label audits and rating ablations; and Person C built our cross-platform transfer benchmark and the production decision-support dashboard you see here."*

---

## 🔍 Step 2: Single Review Screening & Phrase Highlighter (0:45 – 2:15)

Ensure the **Single review** workspace is selected.

### Case A: Clearly Genuine Review (Low Risk < 20%)
1. **Copy & Paste:**
   ```text
   I used the non-stick frying pan three times this week. It heats evenly across the surface but the handle gets slightly warm after fifteen minutes on high heat.
   ```
2. Set Star Rating to **4 ★** and click **Analyze review**.
3. **Expected Score:** Raw Risk: **6.9%** | Calibrated Risk: **8.2%** | Decision: **Low Risk · Automated Approval** (Green).
4. **Talking Point:**
   > *"Notice that realistic product observations, balanced tone, and moderate length pull the SHAP log-odds negative, resulting in a safe low-risk rating eligible for automated approval."*

### Case B: Borderline Short 5★ Review (Length vs. Rating Trade-off)
1. **Copy & Paste:**
   ```text
   AMAZING PRODUCT!!! BEST PURCHASE EVER 10/10 MUST BUY NOW GO GO GO!!!!
   ```
2. Set Star Rating to **5 ★** and click **Analyze review**.
3. **Expected Score:** Raw Risk: **34.2%** | Calibrated Risk: **35.1%** | Decision: **Moderate Risk · Watchlist** (Amber).
4. **Talking Point:**
   > *"Here is a crucial trade-off discovered in Person B's audit: the 5-star rating adds +1.8 log-odds toward fraud, but the short length (67 characters) subtracts -0.4 log-odds because short reviews have lower empirical fraud rates unless accompanied by explicit purchase disclosures. The system correctly triages this to Moderate Risk rather than triggering a false alarm."*

### Case C: High-Risk Manipulated Review (Flags ≥ 37%)
1. **Copy & Paste:**
   ```text
   I bought this item after reading genuine reviews. Honest review, not sponsored, I actually bought this with my own money. Highly recommended, worth every penny, life changing, must buy!
   ```
2. Set Star Rating to **5 ★** and click **Analyze review**.
3. **Expected Score:** Raw Risk: **85.3%** | Calibrated Risk: **81.7%** | Decision: **High Risk · Inspection Required** (Red).
4. **Point to the Interactive Phrase Highlighter:**
   > *"Look at the real-time Highlighter box: the model detects 5 purchase declarations ('honest review', 'not sponsored', 'actually bought') in red, and 4 promotional phrases ('must buy', 'highly recommended') in amber. These tokens drive SHAP contributions sharply positive, pushing calibrated risk to 81.7%."*

---

## ⚡ Step 3: What-If Star-Rating Simulator (2:15 – 3:00)

1. Keep Case C review text in the box and expand **"⚡ What-If Simulator: Star Rating Sensitivity"**.
2. **Talking Point:**
   > *"Moderators often need counterfactual explainability: 'What if the user had rated this 1 star or 3 stars instead of 5 stars?' Notice the interactive Altair bar chart: at 1★, 2★, 3★, and 4★, the risk stays below 12% (well below our dashed red cutoff). Only at 5★ does the combination of extreme rating and purchase disclosure trigger a fraud alert. This proves the model learns a non-linear interaction between rating extremity and text semantics."*

---

## 📁 Step 4: CSV Batch Triage & Audit Export (3:00 – 3:45)

1. Switch to the **CSV batch** workspace at the top.
2. Drag and drop **`demo/sample_reviews.csv`** into the file uploader.
3. Click **Score reviews**.
4. **Point to the KPI Cards:**
   * **Scored:** 10
   * **High Risk 🚩:** 4
   * **Moderate ⚠️:** 3
   * **Low Risk ✅:** 3
   * **Invalid rows:** 0
5. Click **Download enriched CSV** and open it:
   > *"Notice our enriched triage export: every review is output with its Raw Score, Calibrated Probability, Triage Tier, and the primary signed SHAP factor (e.g. 'Star rating (+)', 'Purchase declarations (+)'). This provides an end-to-end audit trail for trust and safety operations."*

---

## 🔬 Step 5: Evaluation & Scientific Benchmarks (3:45 – 5:00)

Switch to the **Evaluation** workspace.

### 1. Standardized 37% Confusion Matrices
* Show the **"Holdout Confusion Matrices (Threshold = 37%)"** section.
* Select **"Earlier Product History (Prospective)"**:
  * **Amazon Holdout (74,192 reviews):** TP = 10,144 | TN = 57,512 | FP = 1,714 | FN = 4,822 | Precision = **85.5%** | Recall = **67.8%** | F1 = **0.756** | FPR = **2.9%** | AP = **0.822**.
  * **Yelp Transfer (358,957 reviews):** Precision = **12.1%** | Recall = **44.8%** | AP = **0.122**.

### 2. Scientific Feature Sensitivity Audit
* Point to the **Ablation & Outlier Sensitivity Table**:
  > *"Our feature selection was driven by mathematical evidence rather than guesswork:*
  > *1. **The 3-Product Anomaly:** Unclipped `prod_review_count` flagged all 4,162 reviews on 3 outlier Amazon products, collapsing baseline holdout AP from 0.835 to 0.517. Omitting it (`full_44`) or enforcing causal historical dates (`earlier_history`) permanently cures this (0.822 AP).*
  > *2. **Rating-Family Signal Independence:** Removing all 13 rating-derived features (`minus_family_32`) incurs only a -0.042 AP loss, proving textual semantics carry independent manipulation signals.*
  > *3. **Cross-Platform Transfer Collapse:** AP drops from 0.822 on Amazon to 0.122 on Yelp due to ground-truth mismatch (Amazon platform removals vs. Yelp recommendation filter) and domain shift from physical products to restaurants."*

---

## 🏁 Conclusion & Q&A Wrap-Up
> *"To conclude: PeWork demonstrates that review fraud screening requires strict temporal causality, probability calibration, and explainable decision support. Our live dashboard bridges complex ensemble models into actionable moderator workflows. Thank you, and we welcome your questions."*
