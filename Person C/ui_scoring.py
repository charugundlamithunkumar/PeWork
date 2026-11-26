"""Score review text using the frozen v2 app bundle."""

import csv
import html
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "ui_assets"
MAX_TEXT = 20_000
MAX_ROWS = 5_000

# Also support running app.py from inside the Person C folder.
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
features = import_module("Person A.review_pipeline.features")
models = import_module("Person A.review_pipeline.models")
digest = import_module("Person A.review_pipeline.io").digest

LABELS = {
    "rating": "Star rating", "EXT": "Extreme star rating",
    "char_count": "Character count", "word_count": "Word count",
    "avg_word_len": "Average word length", "lexical_diversity": "Vocabulary variety",
    "vader_compound": "Text sentiment", "readability_score": "Reading ease",
    "sentiment_mismatch": "Rating–sentiment gap", "uppercase_ratio": "Uppercase share",
    "specificity_proxy": "Digit share", "first_person_ratio": "First-person wording",
    "digit_count": "Digit count", "elongation_count": "Repeated characters",
    "purchase_declaration_count": "Purchase declarations",
}

# Empirical validation deciles for score calibration
RAW_CAL_POINTS = np.array([0.0, 0.015, 0.025, 0.033, 0.042, 0.053, 0.068, 0.095, 0.208, 0.875, 0.996, 1.0])
OBS_CAL_POINTS = np.array([0.03, 0.0389, 0.0379, 0.0438, 0.0484, 0.0631, 0.0817, 0.1107, 0.1969, 0.8378, 0.5580, 0.60])


def calibrate_score(raw_score):
    """Map raw model probability to empirical calibrated risk based on validation deciles."""
    return float(np.interp(np.clip(raw_score, 0.0, 1.0), RAW_CAL_POINTS, OBS_CAL_POINTS))


def get_risk_tier(score, threshold=0.3697):
    """Categorize risk for triage decision making."""
    if score >= threshold:
        return "High Risk", "Inspection Required", "#D9381E", "#FDE8E8"
    elif score >= 0.20:
        return "Moderate Risk", "Watchlist", "#D9822B", "#FEF3C7"
    else:
        return "Low Risk", "Automated Approval", "#137333", "#DEF7EC"


def analyze_text_spans(text):
    """Identify key phrases (promo, purchase declarations, slang, elongation) in the review."""
    lower = text.lower()
    spans = []

    # 1. Purchase declarations (Strongest signal for fake review)
    for p in getattr(features, "PURCHASE", ()):
        start = 0
        while True:
            idx = lower.find(p, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(p), "purchase", "Purchase declaration", "#991B1B", "#FEE2E2"))
            start = idx + len(p)

    # 2. Promo phrases
    for p in getattr(features, "PROMO", ()):
        start = 0
        while True:
            idx = lower.find(p, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(p), "promo", "Promotional phrase", "#92400E", "#FEF3C7"))
            start = idx + len(p)

    # 3. Slang words
    slang_words = getattr(features, "SLANG", set())
    for match in re.finditer(r"\b\w+\b", lower):
        if match.group() in slang_words:
            spans.append((match.start(), match.end(), "slang", "Slang / Informal", "#581C87", "#F3E8FF"))

    # 4. Repeated character elongation (3+ identical characters)
    for match in re.finditer(r"(.)\1{2,}", text):
        spans.append((match.start(), match.end(), "elongation", "Character repetition", "#1E40AF", "#DBEAFE"))

    # Resolve overlapping spans (prefer earlier, then longer)
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    non_overlapping = []
    last_end = 0
    for s in spans:
        if s[0] >= last_end:
            non_overlapping.append(s)
            last_end = s[1]

    # Build highlighted HTML
    html_pieces = []
    curr = 0
    detected_counts = {}
    for start, end, category, label, color, bg in non_overlapping:
        if start > curr:
            html_pieces.append(html.escape(text[curr:start]))
        snippet = html.escape(text[start:end])
        html_pieces.append(
            f'<mark style="background-color: {bg}; color: {color}; font-weight: 600; '
            f'padding: 2px 6px; border-radius: 4px; border: 1px solid {color}55;" '
            f'title="{label}">{snippet} <small style="font-size:0.75em; opacity:0.85;">[{label}]</small></mark>'
        )
        detected_counts[label] = detected_counts.get(label, 0) + 1
        curr = end
    if curr < len(text):
        html_pieces.append(html.escape(text[curr:]))

    return "".join(html_pieces), detected_counts


def load_model(directory=ASSETS):
    spec = json.loads((directory / "selected_app_with_rating.json").read_text(encoding="utf-8"))
    weights = directory / spec["weights"]
    if digest(weights) != spec["weights_sha256"]:
        raise ValueError("The weights checksum does not match the frozen model.")
    if spec["extractor"] != features.EXTRACTOR_VERSION or spec["kind"] != "xgb":
        raise ValueError("The model and text extractor are incompatible.")
    if spec["calibration"]["method"] != "uncalibrated" or not 0 <= spec["threshold"] <= 1:
        raise ValueError("The app bundle has unsupported calibration or threshold settings.")
    model = XGBClassifier(device="cpu", n_jobs=4)
    model.load_model(weights)
    if model.get_booster().feature_names != spec["features"]:
        raise ValueError("The frozen model feature order does not match its manifest.")
    return model, spec


def review_features(text, rating):
    if len(text) > MAX_TEXT:
        raise ValueError(f"Review text exceeds {MAX_TEXT:,} characters.")
    if str(rating).strip() not in ("1", "2", "3", "4", "5"):
        raise ValueError("Use a whole-number rating from 1 to 5.")
    return features.extract_review(text, int(rating))


def score_review(bundle, text, rating):
    model, spec = bundle
    data = pd.DataFrame([review_features(text, rating)])
    columns = spec["features"]
    score = float(models.predict(model, data, "xgb", columns)[0])
    contributions = models.shap_values(model, data, columns)[0, :-1]
    factors = pd.DataFrame({
        "Feature": [LABELS.get(name, name) for name in columns],
        "Value": data.loc[0, columns].to_numpy(dtype=float),
        "Contribution": contributions,
    })
    factors = factors.loc[factors.Contribution.abs() > 1e-8]
    factors = factors.sort_values("Contribution", key=abs, ascending=False).head(3)
    return score, spec["threshold"], factors, data


def ranking_details(data, references):
    """Apply the frozen v2 scoring equations to the review's criterion values."""
    scores, breakdown = [], {}
    for item in references.values():
        ref = item["reference"]
        columns = ref["features"]
        values = data.loc[0, columns].to_numpy(dtype=float)
        lo, hi, weights = map(np.array, (ref["lo"], ref["hi"], ref["weights"]))
        scaled = np.clip((values - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)
        regret = np.where(ref["benefit"], 1 - scaled, scaled) * weights
        method = item["method"].upper()
        if method == "TOPSIS":
            ideal, anti_ideal = np.linalg.norm(regret), np.linalg.norm(weights - regret)
            score = anti_ideal / (ideal + anti_ideal) if ideal + anti_ideal > 0 else 0.5
            values_for_table = {"Squared distance to ideal": regret**2}
        else:
            def relative(value, bounds):
                low, high = bounds
                return np.clip((value - low) / (high - low), 0, 1) if high > low else 0.5

            utility = relative(regret.sum(), ref["utility_range"])
            worst = relative(regret.max(), ref["regret_range"])
            score = 1 - (item["v"] * utility + (1 - item["v"]) * worst)
            values_for_table = {"Weighted regret": regret}
        scores.append({"Method": method, "Score": float(score), "Threshold": item["threshold"],
                       "Flagged": bool(score >= item["threshold"])})
        breakdown.update(values_for_table)
    details = pd.DataFrame({"Criterion": [LABELS.get(name, name) for name in columns],
                            **breakdown})
    return pd.DataFrame(scores), details


def what_if_ratings(bundle, text):
    """Compute predicted scores and calibration across all 5 star ratings for this review."""
    model, spec = bundle
    columns = spec["features"]
    records = [review_features(text, r) for r in (1, 2, 3, 4, 5)]
    df = pd.DataFrame(records)
    scores = models.predict(model, df, "xgb", columns)
    results = []
    for r, score in zip((1, 2, 3, 4, 5), scores):
        calibrated = calibrate_score(score)
        results.append({
            "Rating": f"{r} ★",
            "Star": r,
            "Risk score": float(score),
            "Calibrated risk": float(calibrated),
            "Flagged": bool(score >= spec["threshold"]),
        })
    return pd.DataFrame(results)


def score_csv(bundle, payload):
    """Return an enriched score, risk tier, and top diagnostic factor for each CSV record."""
    rows = csv.reader(StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    header = [name.strip().lower() for name in next(rows, [])]
    if header.count("review_text") != 1 or header.count("rating") != 1:
        raise ValueError("The CSV needs one review_text column and one rating column.")
    text_col, rating_col = header.index("review_text"), header.index("rating")
    results, values, positions = [], [], []
    model, spec = bundle
    for number, row in enumerate((row for row in rows if row), start=1):
        if number > MAX_ROWS:
            raise ValueError(f"Upload a CSV with {MAX_ROWS:,} reviews or fewer.")
        result = {
            "Row": number,
            "Rating": None,
            "Risk score": np.nan,
            "Calibrated risk": np.nan,
            "Risk tier": "Invalid",
            "Top factor": "",
            "Threshold": spec["threshold"],
            "Decision": "Invalid input",
            "Issue": "",
        }
        results.append(result)
        if len(row) != len(header):
            result["Issue"] = "Column count differs from the header."
            continue
        try:
            extracted = review_features(row[text_col], row[rating_col])
        except ValueError as error:
            result["Issue"] = str(error)
            continue
        result["Rating"] = extracted["rating"]
        positions.append(number - 1)
        values.append(extracted)
    if not results:
        raise ValueError("The CSV contains no review rows.")
    if values:
        df_val = pd.DataFrame(values)
        scores = models.predict(model, df_val, "xgb", spec["features"])
        shap_vals = models.shap_values(model, df_val, spec["features"])[:, :-1]
        for idx, (position, score) in enumerate(zip(positions, scores)):
            calibrated = calibrate_score(score)
            tier_name, _, _, _ = get_risk_tier(score, spec["threshold"])

            row_shaps = shap_vals[idx]
            max_idx = int(np.argmax(np.abs(row_shaps)))
            top_feat_name = LABELS.get(spec["features"][max_idx], spec["features"][max_idx])
            top_sign = "+" if row_shaps[max_idx] > 0 else "-"

            results[position]["Risk score"] = float(score)
            results[position]["Calibrated risk"] = float(calibrated)
            results[position]["Risk tier"] = tier_name
            results[position]["Top factor"] = f"{top_feat_name} ({top_sign})"
            results[position]["Decision"] = "Flag for review" if score >= spec["threshold"] else "Below threshold"
    return pd.DataFrame(results)

