import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
# NO TENSORFLOW IMPORT HERE!
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Network Anomaly Guard", page_icon="🛡️", layout="wide")

ISO_FOREST_PATH = "models/kdd_isolation_forest_fixed.pkl"
# Note: Loading the sklearn pickle file now
AUTOENCODER_PATH = "models/autoencoder_sklearn.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.npy"

@st.cache_resource
def load_iso_forest():
    if os.path.exists(ISO_FOREST_PATH):
        return joblib.load(ISO_FOREST_PATH)
    return None

@st.cache_resource
def load_autoencoder():
    # Load using joblib (Safe for Mac)
    model = joblib.load(AUTOENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = np.load(THRESHOLD_PATH)
    return model, scaler, threshold

@st.cache_data
def load_sample_data():
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    labels = df['labels'].copy()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    X = df.drop('labels', axis=1)
    return X, labels

st.title("🛡️ Network Anomaly Detection System")

# SIDEBAR
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Select Detection Engine:", ("Isolation Forest (Statistical)", "Autoencoder (Deep Learning)"))
st.sidebar.markdown("---")
sensitivity = st.sidebar.select_slider("Sensitivity Level", options=["Low", "Medium", "High"], value="Medium")
sensitivity_map = {"Low": 1.5, "Medium": 1.0, "High": 0.8}
multiplier = sensitivity_map[sensitivity]

# LOAD
X_data, y_labels = load_sample_data()
if 'history' not in st.session_state: st.session_state.history = []

col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_normal = col2.empty()
metric_attack = col3.empty()
status_indicator = col4.empty()

chart_col, log_col = st.columns([2, 1])
with chart_col:
    chart_placeholder = st.empty()
with log_col:
    log_placeholder = st.empty()

if st.button("▶️ Start Live Simulation"):
    if "Isolation" in model_choice:
        model = load_iso_forest()
    else:
        model, scaler, base_threshold = load_autoencoder()
        effective_threshold = base_threshold * multiplier

    for i in range(50):
        idx = np.random.randint(0, len(X_data))
        row = X_data.iloc[[idx]]
        actual = y_labels.iloc[idx]

        if "Isolation" in model_choice:
            score = model.decision_function(row)[0]
            iso_cutoff = 0.0 + (0.05 if sensitivity == "High" else -0.05 if sensitivity == "Low" else 0.0)
            is_anomaly = score < iso_cutoff
        else:
            row_scaled = scaler.transform(row)
            # Scikit-Learn predict() works same as Keras here
            reconstruction = model.predict(row_scaled)
            loss = np.mean(np.power(row_scaled - reconstruction, 2))
            score = -loss
            is_anomaly = loss > effective_threshold

        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        st.session_state.history.append({"Type": actual, "Score": round(score, 4), "Status": status})
        
        df_hist = pd.DataFrame(st.session_state.history)
        anomalies = len(df_hist[df_hist['Status'] == "🚨 ANOMALY"])
        
        metric_total.metric("Packets", len(df_hist))
        metric_normal.metric("Normal", len(df_hist) - anomalies)
        metric_attack.metric("Threats", anomalies)
        
        if is_anomaly: status_indicator.error(f"THREAT: {actual}")
        else: status_indicator.success("SECURE")
            
        chart_placeholder.line_chart(df_hist['Score'].tail(50))
        log_placeholder.dataframe(df_hist[['Type', 'Status']].tail(8), height=300)
        time.sleep(0.05)
