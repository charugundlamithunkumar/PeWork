"""Local Streamlit interface for the review screening project."""

import csv
from hashlib import sha256
import json

import altair as alt
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError

from ui_scoring import (
    ASSETS, HERE, MAX_ROWS, MAX_TEXT,
    analyze_text_spans, calibrate_score, get_risk_tier,
    load_model, ranking_details, score_csv, score_review, what_if_ratings
)

EXAMPLES = {
    "Ordinary review": ("I used the pan twice this week. It heats evenly but the handle gets warm.", 4),
    "Ambiguous review": ("Works fine. Amazing quality for the price!", 4),
    "Five stars with neutral text": ("The box contained the item shown in the listing.", 5),
}
COHORTS = {
    "amazon_holdout_all": "Amazon · All held-out products",
    "amazon_holdout_new_reviewers": "Amazon · New reviewers",
    "amazon_holdout_minus3": "Amazon · Three products excluded",
    "amazon_holdout_minus3_new": "Amazon · Three products excluded, new reviewers",
    "yelp_all": "Yelp · All reviews",
    "yelp_inflation": "Yelp · Inflation subset",
}
MODELS = {
    "earlier_history": "Earlier product history",
    "full_45_frozen": "Week 1 · 45 features",
    "full_44_no_prod_review_count": "44 features · No product count",
    "no_family_32": "32 features · No rating family",
    "no_family_31_no_prod_review_count": "31 features · No rating family or product count",
    "rating_only_table": "Rating only",
}
MODEL_ERRORS = (OSError, KeyError, XGBoostError)


@st.cache_resource(show_spinner=False)
def app_bundle():
    return load_model()


@st.cache_data(show_spinner=False)
def app_references():
    return json.loads((ASSETS / "ranking_references.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def evaluation_table():
    return pd.read_csv(HERE / "transfer_results.csv")


def use_example():
    text, rating = EXAMPLES[st.session_state.example]
    st.session_state.update(review_text=text, review_rating=rating, review_result=None)


def show_review():
    left, right = st.columns([1, 1.1], gap="medium")
    with left:
        with st.form("review_form", border=True):
            st.subheader("Review details")
            text = st.text_area("Review text", height=190, max_chars=MAX_TEXT,
                                placeholder="Paste an English product review", key="review_text")
            rating = st.select_slider("Star rating", options=[1, 2, 3, 4, 5],
                                      format_func=lambda value: f"{value} ★", key="review_rating")
            submitted = st.form_submit_button("Analyze review", type="primary", width="stretch",
                                              icon=":material/search:")
        with st.container(border=True):
            example, action = st.columns([1.6, 1], vertical_alignment="bottom")
            example.selectbox("Try an example", list(EXAMPLES), key="example")
            action.button("Use example", on_click=use_example, width="stretch")

        if text.strip():
            highlighted, detected = analyze_text_spans(text)
            with st.container(border=True):
                st.markdown("**Highlighted phrase analysis**")
                if detected:
                    badges = " ".join(
                        f'<span style="background-color:#F3F4F6; color:#374151; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #E5E7EB;">{count}× {label}</span>'
                        for label, count in detected.items()
                    )
                    st.markdown(f'<div style="margin-bottom:8px;">{badges}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background-color:#FAFAFA; border:1px solid #E5E7EB; border-radius:6px; padding:10px; font-size:0.9rem; line-height:1.6; max-height:180px; overflow-y:auto;">{highlighted}</div>',
                    unsafe_allow_html=True
                )
                st.caption("Colored tags mark patterns learned by the screening pipeline (purchase declarations, promo keywords, repeated characters).")

    with right, st.container(border=True):
        st.subheader("Screening result")
        if submitted:
            st.session_state.review_result = None
            if not text.strip():
                st.error("Enter review text to run the analysis.")
            else:
                try:
                    st.session_state.review_result = score_review(app_bundle(), text, rating)
                except MODEL_ERRORS + (ValueError,):
                    st.error("The saved screening model could not be loaded or scored.", title="Model unavailable")
        result = st.session_state.review_result
        if result is None:
            st.caption("Analyze a review to see its score, the flag threshold and the features that moved the score.")
            st.info("This model uses review text and star rating. Product-history results are available in Evaluation.",
                    icon=":material/info:")
            return

        score, threshold, factors, data = result
        calibrated = calibrate_score(score)
        tier_title, tier_action, tier_color, tier_bg = get_risk_tier(score, threshold)

        risk, cal_col, cutoff = st.columns(3)
        risk.metric("Raw risk score", f"{score:.1%}")
        cal_col.metric("Calibrated risk", f"{calibrated:.1%}", help="Empirical probability calibrated on holdout deciles")
        cutoff.metric("Flag threshold", f"{threshold:.1%}")

        st.markdown(
            f'<div style="background-color:{tier_bg}; border-left: 4px solid {tier_color}; color:{tier_color}; padding:8px 12px; border-radius:4px; font-weight:600; margin-bottom:12px;">'
            f'{tier_title} · {tier_action}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("**Top score contributors**")
        if factors.empty:
            st.caption("No feature had a measurable contribution for this review.")
        else:
            chart = alt.Chart(factors).mark_bar(cornerRadius=3).encode(
                x=alt.X("Contribution:Q", title="Change in raw log-odds"),
                y=alt.Y("Feature:N", sort=factors.Feature.tolist(), title=None),
                color=alt.condition(alt.datum.Contribution > 0, alt.value("#B46736"), alt.value("#248265")),
                tooltip=["Feature:N", alt.Tooltip("Value:Q", format=".3g"),
                         alt.Tooltip("Contribution:Q", format="+.3f")],
            ).properties(height=108)
            st.altair_chart(chart, width="stretch", alt="Three features with the largest signed model contributions")
            st.caption("Positive bars raise the score. Negative bars lower it. Values are shown in raw log-odds.")

        with st.expander("⚡ What-If Simulator: Star Rating Sensitivity", expanded=False):
            st.caption("How would the risk score change if the reviewer had chosen a different star rating for this text?")
            try:
                wi_data = what_if_ratings(app_bundle(), text)
                wi_chart = alt.Chart(wi_data).mark_bar(cornerRadius=3).encode(
                    x=alt.X("Rating:N", sort=["1 ★", "2 ★", "3 ★", "4 ★", "5 ★"], title=None),
                    y=alt.Y("Risk score:Q", title="Risk score", scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition(alt.datum["Risk score"] >= threshold, alt.value("#D9381E"), alt.value("#248265")),
                    tooltip=["Rating:N", alt.Tooltip("Risk score:Q", format=".1%"), alt.Tooltip("Calibrated risk:Q", format=".1%")]
                ).properties(height=130)
                cutoff_line = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
                    color="#D9381E", strokeDash=[4, 4]).encode(y="threshold:Q")
                st.altair_chart(wi_chart + cutoff_line, width="stretch")
            except Exception:
                st.caption("What-If simulator requires valid model weights.")

        with st.expander("TOPSIS and VIKOR comparisons"):
            try:
                rankings, criteria = ranking_details(data, app_references())
            except (OSError, ValueError, KeyError):
                st.info("The saved ranking references are unavailable.")
            else:
                st.caption("These are separate ranking scores using frozen Amazon development references.")
                st.dataframe(rankings, hide_index=True, alt="TOPSIS and VIKOR scores and frozen thresholds",
                             column_config={key: st.column_config.NumberColumn(format="%.3f")
                                            for key in ("Score", "Threshold")})
                st.dataframe(criteria, hide_index=True, alt="Feature-level distance and regret by criterion")


def show_batch():
    st.subheader("Score a CSV")
    st.caption(f"Use columns `review_text` and `rating`. Upload UTF-8 CSV files with up to {MAX_ROWS:,} reviews. "
               "The results include calibrated probabilities, risk tiers, and primary diagnostic factors.")
    st.download_button("Download CSV template", data="review_text,rating\nThe item arrived as described.,4\n",
                       file_name="review_template.csv", mime="text/csv", icon=":material/download:")
    with st.container(border=True):
        upload = st.file_uploader("Review CSV", type="csv", key="review_csv")
        score = st.button("Score reviews", type="primary", disabled=upload is None, key="score_batch")
        if upload is None:
            st.session_state.batch_result = None
            return
        payload = upload.getvalue()
        fingerprint = sha256(payload).hexdigest()
        if score:
            st.session_state.batch_result = None
            try:
                with st.spinner("Scoring review texts…"):
                    st.session_state.batch_result = (fingerprint, score_csv(app_bundle(), payload))
            except (csv.Error, UnicodeError, ValueError) as error:
                st.error(str(error), title="Check the CSV")
            except MODEL_ERRORS:
                st.error("The saved screening model could not be loaded.", title="Model unavailable")
        result = st.session_state.batch_result
        if result is None or result[0] != fingerprint:
            return
        frame = result[1]
        scored_col, flagged_col, mod_col, low_col, invalid_col = st.columns(5)
        scored_col.metric("Scored", f"{frame['Risk score'].count():,}", border=True)
        flagged_col.metric("High Risk 🚩", f"{frame['Risk tier'].eq('High Risk').sum():,}", border=True)
        mod_col.metric("Moderate ⚠️", f"{frame['Risk tier'].eq('Moderate Risk').sum():,}", border=True)
        low_col.metric("Low Risk ✅", f"{frame['Risk tier'].eq('Low Risk').sum():,}", border=True)
        invalid_col.metric("Invalid rows", f"{frame.Decision.eq('Invalid input').sum():,}", border=True)

        st.dataframe(frame, hide_index=True, alt="Batch results with scores, tiers, and top factors",
                     column_config={
                         "Risk score": st.column_config.NumberColumn(format="%.1f%%"),
                         "Calibrated risk": st.column_config.NumberColumn(format="%.1f%%"),
                         "Threshold": st.column_config.NumberColumn(format="%.1f%%"),
                     })
        st.download_button("Download enriched CSV", frame.to_csv(index=False), "scored_reviews_enriched.csv", "text/csv",
                           icon=":material/download:")



def show_evaluation():
    st.subheader("Held-out model results")
    st.caption("Compare the six team models and references. These results were measured at frozen Amazon cutoffs.")
    try:
        data = evaluation_table()
    except (OSError, ValueError):
        st.error("The team transfer results could not be loaded.")
        return
    cohort_col, model_col = st.columns([1, 1])
    cohort = cohort_col.selectbox("Test cohort", list(COHORTS), format_func=COHORTS.__getitem__, key="cohort")
    model = model_col.selectbox("Model", list(MODELS), format_func=MODELS.__getitem__, key="eval_model")
    rows = data[data.test_set.eq(cohort)]
    row = rows.loc[rows.model.eq(model)].iloc[0]
    if cohort.startswith("yelp"):
        st.info("Yelp measures restaurant reviews filtered by the platform. It uses a different outcome from Amazon.",
                icon=":material/info:")
    if "minus3" in cohort:
        st.caption("The three products were selected after inspecting holdout errors. This is a sensitivity check.")
    for col, label, value in zip(st.columns(4), ["Reviews", "Positive rate", "Average precision", "ROC-AUC"],
                                 [f"{row.n:,}", f"{row.prevalence:.1%}", f"{row.ap:.3f}", f"{row.roc_auc:.3f}"]):
        col.metric(label, value, border=True)
    compare, detail = st.columns([1.25, 1], gap="medium")
    with compare, st.container(border=True):
        st.markdown("**Average precision across models**")
        plot = rows.assign(Model=rows.model.map(MODELS), Selected=rows.model.eq(model))
        bars = alt.Chart(plot).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("ap:Q", title="Average precision", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Model:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Selected, alt.value("#18644C"), alt.value("#9DB8AA")),
            tooltip=["Model:N", alt.Tooltip("ap:Q", title="Average precision", format=".3f")],
        ).properties(height=208)
        baseline = alt.Chart(pd.DataFrame({"prevalence": [row.prevalence]})).mark_rule(
            color="#525E57", strokeDash=[4, 4]).encode(x="prevalence:Q")
        st.altair_chart(bars + baseline, width="stretch",
                        alt="Average precision by model with a dashed line for the cohort's positive rate")
        st.caption("The dashed line marks the cohort’s positive rate.")
    with detail, st.container(border=True):
        st.markdown("**Decisions at the frozen threshold**")
        for col, label, value in zip(st.columns(3), ["Precision", "Recall", "F1"],
                                     [f"{row.precision:.1%}", f"{row.recall:.1%}", f"{row.f1:.3f}"]):
            col.metric(label, value)
        confusion = pd.DataFrame({
            "Actual label": ["Negative", "Positive"],
            "Predicted negative": [row.tn, row.fn],
            "Predicted positive": [row.fp, row.tp],
        })
        st.dataframe(confusion, hide_index=True, alt="Confusion matrix for the selected model and test cohort")
        st.caption(f"Threshold {row.threshold:.4f} · Trapezoidal PR-AUC {row.pr_auc:.3f}")

    with st.expander("🔬 Research Analysis: Cross-Platform Transfer Dynamics (Amazon vs Yelp)", expanded=False):
        st.markdown(
            """
            **Why does average precision drop from ~0.82 on Amazon to ~0.12 on Yelp?**
            1. **Target Formulation Discrepancy:**
               - **Amazon Ground Truth:** Defined by seller 5-star manipulation heuristics combined with documented platform removals.
               - **Yelp Ground Truth:** Yelp's proprietary recommendation engine filter, which penalizes newer or less active accounts rather than solely malicious fake reviews.
            2. **Feature Coverage & Historical Dependencies:**
               - The best Amazon models (`earlier_history`, `full_45`) rely heavily on prior reviewer rating distributions (`PR`, `NR`) and product volumes. Cross-platform transfer suffers because user history does not bridge across platforms.
            3. **Domain Shift:**
               - Linguistic cues differ between physical consumer products (packaging, delivery, specs) and hospitality/restaurant reviews (service, ambiance, taste).
            """
        )
    st.caption("Live screening uses the separate v2 text-and-rating model. The models shown here use the team's "
               "transfer evaluation and may require product or reviewer history.")


st.set_page_config(page_title="Review screening", page_icon=":material/rule:", layout="wide")
for key in ("review_result", "batch_result"):
    st.session_state.setdefault(key, None)
st.session_state.setdefault("review_rating", 3)

st.title("Review screening")
st.caption("Screen English product reviews for manual inspection and explore how the team’s models performed on Amazon and Yelp.")
view = st.segmented_control("Screening workspace", ["Single review", "CSV batch", "Evaluation"],
                            default="Single review", selection_mode="single", key="workspace",
                            label_visibility="collapsed")
st.space("small")
if view in ("CSV batch", "Evaluation"):
    # Keep the current form values when Streamlit cleans up hidden widget state.
    for key in ("review_text", "review_rating"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
if view == "CSV batch":
    show_batch()
elif view == "Evaluation":
    show_evaluation()
else:
    show_review()
