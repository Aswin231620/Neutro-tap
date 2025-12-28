import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import tempfile
import matplotlib.pyplot as plt
import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NeuroTap – Parkinson Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroTap – Parkinson Detection using Voice")

# ---------------- LOAD MODEL ----------------
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")
EXPECTED_FEATURES = list(scaler.feature_names_in_)

# ---------------- HELPERS ----------------
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
    c.drawString(50, 690, "This report is AI-assisted and not a medical diagnosis.")
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

    # Pad to match training features
    padded = np.zeros(len(EXPECTED_FEATURES))
    padded[:len(mfcc_mean)] = mfcc_mean
    return pd.DataFrame([padded], columns=EXPECTED_FEATURES)

# ---------------- INPUT MODE ----------------
mode = st.radio(
    "Choose input type",
    ["CSV (Extracted Features)", "Audio (.wav)"]
)

# ================= CSV MODE =================
if mode == "CSV (Extracted Features)":
    st.subheader("Upload CSV with Voice Features")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Keep only trained features
        data = data[[c for c in EXPECTED_FEATURES if c in data.columns]]

        scaled = scaler.transform(data)
        probas = model.predict_proba(scaled)[:, 1]
        preds = model.predict(scaled)

        data["Prediction"] = ["Parkinson" if p == 1 else "Healthy" for p in preds]
        data["Risk %"] = (probas * 100).round(2)

        st.success("Prediction Complete")
        st.dataframe(data)

# ================= AUDIO MODE =================
else:
    st.subheader("Upload Voice Recording (.wav)")
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        features_df = extract_features(audio_path)
        scaled = scaler.transform(features_df)

        prob = model.predict_proba(scaled)[0][1]
        pred = model.predict(scaled)[0]

        risk, icon = get_risk_level(prob)

        # ---------------- RESULTS ----------------
        st.subheader(f"{icon} Risk Level: {risk}")

        st.write("Prediction Confidence")
        st.progress(int(prob * 100))
        st.write(f"**{prob*100:.2f}%**")

        if prob < 0.3:
            st.success(
                "The voice features are within normal ranges. "
                "No strong indicators of Parkinson’s disease were detected."
            )
        elif prob < 0.6:
            st.warning(
                "Some voice irregularities were detected. "
                "This may indicate early or mild symptoms."
            )
        else:
            st.error(
                "Significant voice abnormalities detected. "
                "High risk of Parkinson’s disease."
            )

        st.info(f"🧠 Estimated Condition: **{disease_stage(prob)}**")

        # ---------------- VISUALIZATION ----------------
        radar_chart(features_df.iloc[0].values[:10])

        # ---------------- PDF ----------------
        pdf = generate_pdf(prob, risk)
        st.download_button(
            "📄 Download Report (PDF)",
            data=pdf,
            file_name="NeuroTap_Report.pdf",
            mime="application/pdf"
        )

# ---------------- FOOTER ----------------
st.caption(
    "⚠️ This application is for research and educational purposes only. "
    "It does not replace professional medical diagnosis."
)
