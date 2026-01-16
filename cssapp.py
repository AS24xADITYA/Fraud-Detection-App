import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ---------------------------------
# CUSTOM CSS (NO IMPORTS NEEDED)
# ---------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

h1, h2, h3 {
    text-align: center;
    color: #f8fafc;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.3em;
    font-weight: bold;
    border: none;
}
.stButton > button:hover {
    transform: scale(1.03);
}

[data-testid="stMetric"] {
    background-color: #020617;
    border: 1px solid #1e293b;
    padding: 1rem;
    border-radius: 12px;
}

.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL & DATA
# ---------------------------------
model = load_model("autoencoder_creditcard.h5", compile=False)
data = pd.read_csv("creditcard.csv")

X = data.drop(["Time", "Class"], axis=1)
y = data["Class"]

THRESHOLD = 1.85

# Store transaction history in session
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# HEADER
# ---------------------------------
st.title("💳 Credit Card Fraud Detection System")
st.caption("Real-time ML-based fraud detection using Autoencoder")
st.success("Model loaded successfully")

st.divider()

# ---------------------------------
# DATASET AUTO-FILL
# ---------------------------------
st.subheader("📂 Test Using Real Dataset Transaction")

index = st.slider(
    "Select Transaction Index",
    0,
    len(X) - 1,
    0
)

transaction = X.iloc[index].values.reshape(1, 29)

if st.button("Check Dataset Transaction"):
    reconstructed = model.predict(transaction)
    error = np.mean(np.square(transaction - reconstructed))

    risk_score = min(100, (error / (THRESHOLD * 2)) * 100)
    prediction = "FRAUD" if error > THRESHOLD else "GENUINE"
    actual = "FRAUD" if y.iloc[index] == 1 else "GENUINE"

    st.metric("Reconstruction Error", f"{error:.4f}")
    st.progress(int(risk_score))
    st.caption(f"Fraud Risk Score: {risk_score:.1f}%")

    if prediction == "FRAUD":
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
        st.warning(
            "Explanation: Transaction behavior significantly deviates "
            "from the learned normal spending patterns."
        )
    else:
        st.success("✅ GENUINE TRANSACTION")
        st.info(
            "Explanation: Transaction closely matches normal customer behavior."
        )

    st.caption(f"Actual Label in Dataset: {actual}")

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Type": "Dataset",
        "Error": round(error, 4),
        "Risk %": round(risk_score, 1),
        "Prediction": prediction
    })

st.divider()

# ---------------------------------
# SIMULATION SECTION
# ---------------------------------
st.subheader("🧪 Simulate Transaction Behavior")

col1, col2 = st.columns(2)

with col1:
    if st.button("Simulate NORMAL Transaction"):
        transaction = np.random.normal(0, 1, 29).reshape(1, 29)
        reconstructed = model.predict(transaction)
        error = np.mean(np.square(transaction - reconstructed))
        risk_score = min(100, (error / (THRESHOLD * 2)) * 100)

        st.metric("Reconstruction Error", f"{error:.4f}")
        st.progress(int(risk_score))
        st.success("✅ NORMAL TRANSACTION")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Type": "Simulated Normal",
            "Error": round(error, 4),
            "Risk %": round(risk_score, 1),
            "Prediction": "GENUINE"
        })

with col2:
    if st.button("Simulate FRAUD Transaction"):
        transaction = np.random.normal(0, 10, 29).reshape(1, 29)
        reconstructed = model.predict(transaction)
        error = np.mean(np.square(transaction - reconstructed))
        risk_score = min(100, (error / (THRESHOLD * 2)) * 100)

        st.metric("Reconstruction Error", f"{error:.4f}")
        st.progress(int(risk_score))
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Type": "Simulated Fraud",
            "Error": round(error, 4),
            "Risk %": round(risk_score, 1),
            "Prediction": "FRAUD"
        })

st.divider()

# ---------------------------------
# TRANSACTION HISTORY PANEL
# ---------------------------------
st.subheader("📊 Transaction History (Current Session)")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download History as CSV",
        csv,
        "transaction_history.csv",
        "text/csv"
    )
else:
    st.caption("No transactions checked yet.")
