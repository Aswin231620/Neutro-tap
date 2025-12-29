import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import tempfile
import matplotlib.pyplot as plt
import io
import os

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="NeuroTap – Parkinson Detection",
    page_icon="🧠",
    layout="centered"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main { background-color: #f8fafc; }
.block-container { padding-top: 2rem; }
.metric-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🧠 NeuroTap")
st.subheader("Parkinson’s Disease Detection using Voice Analysis")
st.caption(
    "An AI‑assisted tool for early Parkinson’s risk screening using voice features."
)
st.divider()

# ================= LOAD MODEL (DEPLOYMENT SAFE) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "parkinson_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()
EXPECTED_FEATURES = list(scaler.feature_names_in_)

# ================= HELPERS =================
def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk", "🟢"
    elif prob < 0.6:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"

def disease_stage(prob):
    if prob < 0.3:
        return "Healthy / No PD"
    elif prob < 0.5:
        return "Possible Early Stage"
    elif prob < 0.75:
        return "Moderate Stage"
    else:
        return "Severe Risk Stage"

def generate_pdf(prob, risk):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 12)

    c.drawString(50, 800, "NeuroTap – Parkinson Voice Analysis Report")
    c.drawString(50, 770, f"Risk Level: {risk}")
    c.drawString(50, 750, f"Prediction Confidence: {prob*100:.2f}%")

    c.drawString(50, 710, "Disclaimer:")
    c.drawString(50, 690, "This report is AI‑assisted and not a medical diagnosis.")
    c.drawString(50, 670, "Consult a neurologist for clinical evaluation.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def radar_chart(features):
    labels = EXPECTED_FEATURES[:len(features)]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.concatenate((features, [features[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=6)
    ax.set_title("Voice Feature Radar")
    st.pyplot(fig)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    padded = np.zeros(len(EXPECTED_FEATURES))
    padded[:len(mfcc_mean)] = mfcc_mean

    return pd.DataFrame([padded], columns=EXPECTED_FEATURES)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Input Settings")
mode = st.sidebar.radio(
    "Choose Input Type",
    ["CSV (Extracted Features)", "Audio (.wav)"]
)

st.sidebar.info(
    "Upload extracted voice features (CSV) "
    "or a raw voice recording (.wav)."
)

# ================= MAIN CONTENT =================
if mode == "CSV (Extracted Features)":
    st.subheader("📄 Upload CSV with Voice Features")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data = data[[c for c in EXPECTED_FEATURES if c in data.columns]]

        scaled = scaler.transform(data)
        probas = model.predict_proba(scaled)[:, 1]
        preds = model.predict(scaled)

        data["Prediction"] = ["Parkinson" if p == 1 else "Healthy" for p in preds]
        data["Risk (%)"] = (probas * 100).round(2)

        st.success("Prediction Completed Successfully")
        st.dataframe(data, width="stretch")

else:
    st.subheader("🎤 Upload Voice Recording (.wav)")
    audio_file = st.file_uploader("Upload WAV File", type=["wav"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        features_df = extract_features(audio_path)
        scaled = scaler.transform(features_df)

        prob = model.predict_proba(scaled)[0][1]
        risk, icon = get_risk_level(prob)

        st.divider()
        st.subheader("📊 Analysis Result")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div class='metric-box'><h4>{icon} Risk Level</h4><h2>{risk}</h2></div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='metric-box'><h4>Confidence</h4><h2>{prob*100:.2f}%</h2></div>",
                unsafe_allow_html=True
            )

        st.progress(int(prob * 100))

        if prob < 0.3:
            st.success("Voice features are within normal range.")
        elif prob < 0.6:
            st.warning("Mild voice irregularities detected. Early screening advised.")
        else:
            st.error("High risk detected. Medical consultation recommended.")

        st.info(f"🧠 Estimated Condition: **{disease_stage(prob)}**")

        st.subheader("📈 Voice Feature Visualization")
        radar_chart(features_df.iloc[0].values[:10])

        pdf = generate_pdf(prob, risk)
        st.download_button(
            "📄 Download Report (PDF)",
            data=pdf,
            file_name="NeuroTap_Report.pdf",
            mime="application/pdf"
        )

# ================= FOOTER =================
st.divider()
st.caption(
    "⚠️ This application is for educational and research purposes only. "
    "It does not replace professional medical diagnosis."
)
