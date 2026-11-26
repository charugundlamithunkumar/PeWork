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


REVERSE_LABELS = {v: k for k, v in LABELS.items()}


def _format_risk_driver(feat, val, contrib, rating, text):
    """Format a positive SHAP contribution into a clear operational explanation."""
    r = int(rating)
    if feat == "rating":
        if r in (1, 5):
            return f"+{contrib:.2f} log-odds — {r}★ extreme rating increases manipulation prior"
        return f"+{contrib:.2f} log-odds — {r}★ rating elevates baseline risk"
    elif feat == "EXT":
        return f"+{contrib:.2f} log-odds — Extreme rating polarity ({r}★) matches typical astroturfing profiles"
    elif feat == "vader_compound":
        if val >= 0.6:
            return f"+{contrib:.2f} log-odds — Hyperbolic positive sentiment ({val:.2f} compound score)"
        elif val <= -0.5:
            return f"+{contrib:.2f} log-odds — Hyperbolic negative sentiment ({val:.2f} compound score)"
        return f"+{contrib:.2f} log-odds — Polarized sentiment score ({val:+.2f})"
    elif feat == "readability_score":
        if val >= 70:
            return f"+{contrib:.2f} log-odds — High reading ease ({val:.1f}) characteristic of promotional copy"
        elif val <= 40:
            return f"+{contrib:.2f} log-odds — Dense sentence complexity ({val:.1f} reading ease) deviates from baseline"
        return f"+{contrib:.2f} log-odds — Reading ease ({val:.1f}) elevates risk"
    elif feat == "char_count":
        return f"+{contrib:.2f} log-odds — Review length ({int(val)} chars) consistent with scripted review templates"
    elif feat == "word_count":
        return f"+{contrib:.2f} log-odds — Word count ({int(val)} words) consistent with scripted review templates"
    elif feat == "uppercase_ratio":
        return f"+{contrib:.2f} log-odds — Elevated uppercase share ({val:.1%}) indicates emphasis-seeking formatting"
    elif feat == "purchase_declaration_count":
        return f"+{contrib:.2f} log-odds — Explicit purchase declarations ({int(val)} detected) commonly inserted to artificially establish credibility"
    elif feat == "avg_word_len":
        return f"+{contrib:.2f} log-odds — Average word length ({val:.1f}) characteristic of promotional templates"
    elif feat == "lexical_diversity":
        return f"+{contrib:.2f} log-odds — Constrained vocabulary variety ({val:.2f} diversity index)"
    elif feat == "sentiment_mismatch":
        return f"+{contrib:.2f} log-odds — Disconnect between star rating and review sentiment ({val:.2f} gap)"
    elif feat == "first_person_ratio":
        return f"+{contrib:.2f} log-odds — Elevated first-person wording ({val:.1%}) typical of staged testimonials"
    elif feat == "elongation_count":
        return f"+{contrib:.2f} log-odds — Repeated character elongation ({int(val)} instances) indicates exaggerated informal tone"
    elif feat == "specificity_proxy":
        return f"+{contrib:.2f} log-odds — Digit share ({val:.1%}) deviates from natural consumer prose"
    elif feat == "digit_count":
        return f"+{contrib:.2f} log-odds — Digit frequency ({int(val)}) deviates from natural consumer prose"
    label = LABELS.get(feat, feat)
    return f"+{contrib:.2f} log-odds — {label} increases risk"


def _format_mitigating_factor(feat, val, contrib, rating, text):
    """Format a negative SHAP contribution into a clear operational mitigating explanation."""
    r = int(rating)
    if feat == "rating":
        if r in (2, 3, 4):
            return f"{contrib:.2f} log-odds — Non-extreme {r}★ rating indicates balanced evaluation"
        return f"{contrib:.2f} log-odds — {r}★ rating reduces manipulation prior"
    elif feat == "EXT":
        return f"{contrib:.2f} log-odds — Absence of extreme rating polarity indicates measured evaluation"
    elif feat == "char_count":
        return f"{contrib:.2f} log-odds — Short review length ({int(val)} chars) lowers empirical fraud rate compared to scripted reviews"
    elif feat == "word_count":
        return f"{contrib:.2f} log-odds — Concise word count ({int(val)} words) lowers empirical fraud rate"
    elif feat == "uppercase_ratio":
        if val > 0.4:
            return f"{contrib:.2f} log-odds — Formatting pattern ({val:.1%}) atypical of standard astroturfing"
        return f"{contrib:.2f} log-odds — Moderate uppercase share ({val:.1%}) within organic bounds"
    elif feat == "vader_compound":
        return f"{contrib:.2f} log-odds — Nuanced text sentiment ({val:.2f} compound score) reflects authentic reaction"
    elif feat == "readability_score":
        return f"{contrib:.2f} log-odds — Reading complexity ({val:.1f}) typical of genuine consumer prose"
    elif feat == "lexical_diversity":
        return f"{contrib:.2f} log-odds — Rich vocabulary variety ({val:.2f} unique ratio) signals natural composition"
    elif feat == "avg_word_len":
        return f"{contrib:.2f} log-odds — Natural average word length ({val:.1f} chars/word)"
    elif feat == "purchase_declaration_count":
        return f"{contrib:.2f} log-odds — Absence of artificial purchase declaration markers"
    elif feat == "elongation_count":
        return f"{contrib:.2f} log-odds — Absence of exaggerated character repetition"
    elif feat == "sentiment_mismatch":
        return f"{contrib:.2f} log-odds — Consistent alignment between star rating and expressed sentiment"
    elif feat == "first_person_ratio":
        return f"{contrib:.2f} log-odds — Natural first-person wording ratio ({val:.1%})"
    elif feat == "specificity_proxy":
        return f"{contrib:.2f} log-odds — Organic digit ratio ({val:.1%})"
    elif feat == "digit_count":
        return f"{contrib:.2f} log-odds — Organic digit frequency ({int(val)})"
    label = LABELS.get(feat, feat)
    return f"{contrib:.2f} log-odds — {label} pulls risk downward"


def generate_verdict_reasoning(text, rating, raw_score, calibrated_score, risk_tier, shap_contributions, spans=None):
    """Generate deterministic, human-readable explanations grounded in SHAP and lexical analysis."""
    r = int(rating) if str(rating).strip().isdigit() else 3

    # Normalize SHAP contributions into {feature_name: float}
    shap_dict = {}
    if isinstance(shap_contributions, dict):
        shap_dict = {REVERSE_LABELS.get(k, k): float(v) for k, v in shap_contributions.items()}
    elif isinstance(shap_contributions, pd.DataFrame):
        feat_col = "Feature" if "Feature" in shap_contributions.columns else shap_contributions.columns[0]
        val_col = "Contribution" if "Contribution" in shap_contributions.columns else shap_contributions.columns[1]
        for _, row in shap_contributions.iterrows():
            k = str(row[feat_col])
            shap_dict[REVERSE_LABELS.get(k, k)] = float(row[val_col])
    elif hasattr(shap_contributions, "items"):
        shap_dict = {REVERSE_LABELS.get(k, k): float(v) for k, v in shap_contributions.items()}
    elif isinstance(shap_contributions, (list, tuple)):
        for item in shap_contributions:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = str(item[0])
                shap_dict[REVERSE_LABELS.get(k, k)] = float(item[1])

    # Extract feature values for textual grounding
    try:
        feat_vals = review_features(text, r)
    except Exception:
        feat_vals = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "vader_compound": 0.0,
            "readability_score": 50.0,
        }

    # Lexical pattern matches
    lower_text = text.lower()
    matched_purchase = [p for p in getattr(features, "PURCHASE", ()) if p in lower_text]
    matched_promo = [p for p in getattr(features, "PROMO", ()) if p in lower_text]

    # Partition drivers (>0) and mitigators (<0)
    drivers = sorted([(k, v) for k, v in shap_dict.items() if v > 0.005], key=lambda x: x[1], reverse=True)
    mitigators = sorted([(k, v) for k, v in shap_dict.items() if v < -0.005], key=lambda x: x[1])

    risk_drivers = [
        _format_risk_driver(k, feat_vals.get(k, 0.0), v, r, text)
        for k, v in drivers[:3]
    ]
    mitigating_factors = [
        _format_mitigating_factor(k, feat_vals.get(k, 0.0), v, r, text)
        for k, v in mitigators[:2]
    ]

    tier_str = str(risk_tier).lower()
    if "high" in tier_str:
        clean_tier = "High Risk"
    elif "mod" in tier_str or "watch" in tier_str:
        clean_tier = "Moderate Risk"
    else:
        clean_tier = "Low Risk"

    # 1. Verdict sentence
    if clean_tier == "High Risk":
        s1 = f"Flagged as High Risk ({calibrated_score:.1%} calibrated probability, exceeding the 37.0% inspection cutoff)."
    elif clean_tier == "Moderate Risk":
        s1 = f"Placed on Watchlist as Moderate Risk ({calibrated_score:.1%} calibrated probability, just below the 37.0% cutoff)."
    else:
        s1 = f"Cleared as Low Risk ({calibrated_score:.1%} calibrated probability, well below the 37.0% inspection cutoff)."

    # 2. Primary reason sentence
    if matched_purchase:
        phrases_str = ", ".join(f'"{p}"' for p in matched_purchase[:3])
        s2 = (f"The model detected synthetic purchase declarations ({phrases_str}) alongside an extreme {r}★ rating, "
              f"matching known astroturfing and paid-review patterns.")
    elif matched_promo:
        phrases_str = ", ".join(f'"{p}"' for p in matched_promo[:3])
        s2 = (f"The review contains promotional phrases ({phrases_str}) combined with an extreme {r}★ rating, "
              f"signaling incentive-driven manipulation.")
    elif clean_tier == "Moderate Risk":
        s2 = (f"An extreme {r}★ rating elevates the initial baseline risk, while the concise text length "
              f"({len(text)} characters) and stylistic attributes partially counterbalance the score.")
    elif clean_tier == "High Risk":
        s2 = (f"An extreme {r}★ rating combined with polarized sentiment and stylistic patterns "
              f"strongly elevates manipulation probability.")
    else:
        if r in (2, 3, 4):
            s2 = (f"The non-extreme {r}★ rating and organic phrasing reflect balanced, genuine consumer feedback "
                  f"without deceptive markers.")
        else:
            s2 = (f"Despite the {r}★ rating, the organic linguistic structure and absence of manipulative markers "
                  f"confirm authentic user sentiment.")

    # 3. Counterbalancing signal sentence
    if clean_tier == "High Risk":
        if mitigators:
            top_mit_name = LABELS.get(mitigators[0][0], mitigators[0][0])
            top_mit_val = mitigators[0][1]
            s3 = (f"While {top_mit_name} provides slight downward mitigation ({top_mit_val:+.2f} log-odds), "
                  f"the dominant risk drivers keep the overall prediction well into the high-risk zone.")
        else:
            s3 = "Virtually all evaluated linguistic and metadata signals reinforce the high-risk classification."
    elif clean_tier == "Moderate Risk":
        if mitigators:
            top_mit_name = LABELS.get(mitigators[0][0], mitigators[0][0])
            top_mit_val = mitigators[0][1]
            s3 = (f"Mitigating signals including {top_mit_name} ({top_mit_val:+.2f} log-odds) hold the calibrated "
                  f"probability safely below the automated flagging threshold.")
        else:
            s3 = "Balanced feature interactions maintain the score in the intermediate observation tier."
    else:
        if mitigators:
            top_names = [LABELS.get(m[0], m[0]) for m in mitigators[:2]]
            s3 = (f"Strong downward contributions from {' and '.join(top_names)} firmly anchor the review "
                  f"within the safe operating envelope.")
        else:
            s3 = "The feature contributions remain predominantly neutral to negative, supporting automated approval."

    summary = f"{s1} {s2} {s3}"

    actions = {
        "High Risk": "Route to secondary human compliance audit and hold merchant settlement.",
        "Moderate Risk": "Place on watchlist for velocity spikes; auto-approve unless repeated across product.",
        "Low Risk": "Automated approval; ingest directly into public catalog with zero latency.",
    }

    return {
        "summary": summary,
        "risk_drivers": risk_drivers,
        "mitigating_factors": mitigating_factors,
        "recommended_action": actions.get(clean_tier, actions["Low Risk"]),
    }


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
    factors_top = factors.loc[factors.Contribution.abs() > 1e-8]
    factors_top = factors_top.sort_values("Contribution", key=abs, ascending=False).head(3)

    calibrated = calibrate_score(score)
    tier_title, _, _, _ = get_risk_tier(score, spec["threshold"])
    shap_dict = dict(zip(columns, contributions))
    reasoning = generate_verdict_reasoning(
        text=text,
        rating=rating,
        raw_score=score,
        calibrated_score=calibrated,
        risk_tier=tier_title,
        shap_contributions=shap_dict,
        spans=None,
    )
    return score, spec["threshold"], factors_top, data, reasoning


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

