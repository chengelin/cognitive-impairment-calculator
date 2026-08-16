from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
FEATURES = ["HSA-L", "PCSA", "BA-EDV", "Education Years", "Age", "CHW-L", "CHW-R", "HCSA-R"]
THRESHOLD = 0.50

META = {
    "Education Years": {"label": "Education Years", "unit": "years", "min": 0.0, "max": 22.0, "p01": 0.0, "p99": 19.0, "default": 9.0, "step": 1.0},
    "Age": {"label": "Age", "unit": "years", "min": 18.0, "max": 100.0, "p01": 41.0, "p99": 90.14, "default": 59.0, "step": 1.0},
    "PCSA": {"label": "Maximum Cross-sectional Area of the Pons (PCSA)", "unit": "cm²", "min": 3.105, "max": 10.05, "p01": 3.6401, "p99": 7.8876, "default": 5.38, "step": 0.01},
    "HSA-L": {"label": "Left Hippocampal Short Axis (HSA-L)", "unit": "mm", "min": 8.63, "max": 21.94, "p01": 9.1502, "p99": 18.738, "default": 13.70, "step": 0.01},
    "BA-EDV": {"label": "Basilar Artery End-diastolic Velocity (BA-EDV)", "unit": "cm/s", "min": 4.97, "max": 50.66, "p01": 9.513, "p99": 44.5704, "default": 21.90, "step": 0.1},
    "CHW-L": {"label": "Left Cerebral Hemisphere Width (CHW-L)", "unit": "mm", "min": 16.64, "max": 76.60, "p01": 57.9912, "p99": 72.0878, "default": 64.41, "step": 0.01},
    "HCSA-R": {"label": "Right Hippocampal Cross-sectional Area (HCSA-R)", "unit": "cm²", "min": 1.18, "max": 5.06, "p01": 1.6156, "p99": 4.168, "default": 2.47, "step": 0.01},
    "CHW-R": {"label": "Right Cerebral Hemisphere Width (CHW-R)", "unit": "mm", "min": 27.33, "max": 78.90, "p01": 57.1492, "p99": 71.869, "default": 64.44, "step": 0.01},
}

GLOBAL_IMPORTANCE = pd.DataFrame(
    {
        "feature": ["Education Years", "Age", "PCSA", "HSA-L", "BA-EDV", "CHW-L", "HCSA-R", "CHW-R"],
        "mean_shap": [0.132139, 0.057210, 0.040909, 0.039694, 0.025041, 0.017848, 0.014564, 0.013491],
    }
)


st.set_page_config(page_title="Cognitive Impairment Risk Calculator", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .hero {padding: 1.35rem 1.5rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg,#073b4c,#087e8b 60%,#3bb6a5); margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2rem; margin: 0 0 .35rem 0;}
    .hero p {margin: 0; opacity: .92;}
    .risk-card {padding: 1.2rem; border-radius: 16px; border: 1px solid #dce8e7; background:#f8fbfb;}
    .risk-number {font-size: 3.1rem; font-weight: 750; line-height: 1; color:#087e8b;}
    .small-note {color:#5b6b70; font-size:.88rem;}
    </style>
    <div class="hero">
      <h1>Cognitive Impairment Risk Calculator</h1>
      <p>Five-fold XGBoost ensemble based on dual-mode transcranial ultrasound (TCS + TCCD)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    bundles = []
    for fold in range(1, 6):
        scaler = joblib.load(MODEL_DIR / f"fold{fold}_scaler.pkl")
        model = joblib.load(MODEL_DIR / f"fold{fold}_xgboost.pkl")
        bundles.append((scaler, model))
    return bundles


def ensemble_predict(raw_frame: pd.DataFrame):
    probabilities = []
    standardized = []
    for scaler, model in load_models():
        z = pd.DataFrame(scaler.transform(raw_frame[FEATURES]), columns=FEATURES)
        probabilities.append(float(model.predict_proba(z)[0, 1]))
        standardized.append(z)
    return float(np.mean(probabilities)), probabilities, standardized


def directional_contributions(standardized_frames):
    values = []
    for (_, model), z in zip(load_models(), standardized_frames):
        estimators = []
        for calibrated in getattr(model, "calibrated_classifiers_", []):
            estimator = getattr(calibrated, "estimator", None)
            if estimator is not None:
                estimators.append(estimator)
        for estimator in estimators:
            contribution = estimator.get_booster().predict(
                __import__("xgboost").DMatrix(z, feature_names=FEATURES), pred_contribs=True
            )[0][:-1]
            values.append(contribution)
    return np.mean(values, axis=0) if values else np.zeros(len(FEATURES))


with st.sidebar:
    st.subheader("Model Performance")
    st.metric("OOF AUC", "0.846", "95% CI 0.799–0.885")
    c1, c2 = st.columns(2)
    c1.metric("Sensitivity", "0.705")
    c2.metric("Specificity", "0.830")
    st.caption("Internal validation: n=387; cognitive impairment: n=129. Classification threshold: 0.50.")
    st.divider()
    st.warning("For research and screening support only. This tool does not replace neuropsychological assessment, diagnostic imaging, or clinical diagnosis.")

left, right = st.columns([0.92, 1.18], gap="large")
with left:
    st.subheader("Patient-specific Inputs")
    values = {}
    display_order = ["Age", "Education Years", "PCSA", "HSA-L", "HCSA-R", "CHW-L", "CHW-R", "BA-EDV"]
    for feature in display_order:
        m = META[feature]
        values[feature] = st.number_input(
            f"{m['label']}（{m['unit']}）",
            min_value=float(m["min"]), max_value=float(m["max"]), value=float(m["default"]),
            step=float(m["step"]), format="%.2f" if m["step"] < 1 else "%.0f",
            help=f"Observed range in the study cohort: {m['min']:g}–{m['max']:g} {m['unit']}",
        )
    calculate = st.button("Calculate Risk", type="primary", use_container_width=True)
    if st.button("Reset to Example Values", use_container_width=True):
        st.rerun()

raw = pd.DataFrame([{name: values[name] for name in FEATURES}])

with right:
    st.subheader("Prediction Result")
    if calculate:
        probability, fold_probs, z_frames = ensemble_predict(raw)
        label = "Higher Risk" if probability >= THRESHOLD else "Lower Risk"
        color = "#c44536" if probability >= THRESHOLD else "#087e8b"
        st.markdown(
            f"<div class='risk-card'><div class='small-note'>Predicted probability of cognitive impairment</div>"
            f"<div class='risk-number' style='color:{color}'>{probability:.1%}</div>"
            f"<h3 style='color:{color};margin-bottom:.2rem'>{label}</h3>"
            f"<div class='small-note'>Risk classification uses the prespecified threshold of {THRESHOLD:.2f}. A higher probability indicates a stronger model prediction of cognitive impairment.</div></div>",
            unsafe_allow_html=True,
        )
        outliers = [META[f]["label"] for f in FEATURES if values[f] < META[f]["p01"] or values[f] > META[f]["p99"]]
        if outliers:
            st.warning("The following inputs fall outside approximately the 1st–99th percentile range of the study cohort; the prediction may be less reliable: " + ", ".join(outliers))

        contribution = directional_contributions(z_frames)
        plot_df = pd.DataFrame({"feature": FEATURES, "value": contribution})
        plot_df["label"] = plot_df["feature"].map(lambda f: META[f]["label"].split("（")[0])
        plot_df = plot_df.sort_values("value")
        fig = go.Figure(go.Bar(
            x=plot_df["value"], y=plot_df["label"], orientation="h",
            marker_color=np.where(plot_df["value"] >= 0, "#d66b5d", "#2b9aa0"),
            hovertemplate="%{y}<br>Directional contribution: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(title="Patient-specific Model Drivers", xaxis_title="Directional contribution (positive: higher risk; negative: lower risk)", yaxis_title="", height=380, margin=dict(l=10,r=10,t=55,b=35))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Prediction Consistency Across Folds"):
            consistency = pd.DataFrame({"Model": [f"Fold {i}" for i in range(1,6)], "Predicted Probability": fold_probs})
            st.dataframe(consistency.style.format({"Predicted Probability": "{:.1%}"}), hide_index=True, use_container_width=True)
            st.caption(f"Range across folds: {min(fold_probs):.1%}–{max(fold_probs):.1%}. The final result is the arithmetic mean of the five probabilities.")
    else:
        st.info("Confirm the eight input variables and select “Calculate Risk”.")
        chart = GLOBAL_IMPORTANCE.sort_values("mean_shap")
        fig = go.Figure(go.Bar(x=chart["mean_shap"], y=chart["feature"], orientation="h", marker_color="#217a8a"))
        fig.update_layout(title="Overall Feature Importance (OOF mean |SHAP|)", xaxis_title="Mean absolute SHAP value", yaxis_title="", height=390, margin=dict(l=10,r=10,t=55,b=35))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("Variable Definitions and Instructions"):
    st.markdown("""
- **PCSA:** maximum cross-sectional area of the pons (cm²).
- **HSA-L:** left hippocampal short axis (mm).
- **HCSA-R:** right hippocampal maximum cross-sectional area (cm²).
- **CHW-L / CHW-R:** left/right cerebral hemisphere width (mm).
- **BA-EDV:** basilar artery end-diastolic velocity (cm/s).
- The model underwent internal five-fold validation in 387 age-matched participants and has not yet undergone independent multicenter external validation.
- The patient-specific contribution plot describes the direction of model effects. The final probability is produced by the calibrated five-fold XGBoost ensemble.
    """)

st.markdown(
    "<div style='text-align:center;color:#66777b;font-size:.92rem;padding:1.4rem 0 .4rem;'>Hangzhou First People’s Hospital</div>",
    unsafe_allow_html=True,
)
