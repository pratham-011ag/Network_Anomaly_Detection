import os
# --- MAC OS FIX: Prevent Threading Deadlock ---
os.environ['OMP_NUM_THREADS'] = '1'
# ----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import LabelEncoder

# --- NEW IMPORT: USE THE CENTRAL DATA LOADER ---
# This ensures Dashboard sees the same data as the Training Script
try:
    from data_loader import load_data
except ImportError:
    # Fallback if running from root directory
    from src.data_loader import load_data

# --- PAGE CONFIG ---
st.set_page_config(page_title="Network Anomaly Guard", page_icon="🛡️", layout="wide")

# --- PATHS ---
ISO_FOREST_PATH = "models/kdd_isolation_forest_fixed.pkl"
AUTOENCODER_PATH = "models/autoencoder_sklearn.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.npy"

# --- LOADERS ---
@st.cache_resource
def load_iso_forest():
    if os.path.exists(ISO_FOREST_PATH):
        return joblib.load(ISO_FOREST_PATH)
    return None

@st.cache_resource
def load_autoencoder():
    model = joblib.load(AUTOENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = np.load(THRESHOLD_PATH)
    return model, scaler, threshold

@st.cache_data
def load_sample_data():
    # CALL THE CENTRAL LOADER (Fixes the Feature Mismatch)
    df = load_data() 
    
    # Split into Features (X) and Labels (y)
    # The loader returns 'is_anomaly' as 0 or 1
    y = df['is_anomaly'].apply(lambda x: "Normal" if x == 0 else "Anomaly")
    X = df.drop('is_anomaly', axis=1)
    
    return X, y

# --- MAIN APP UI ---
st.title("🛡️ Network Anomaly Detection System")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Detection Engine:",
    ("Isolation Forest (Statistical)", "Autoencoder (Deep Learning)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Sensitivity Level")
sensitivity = st.sidebar.select_slider(
    "Adjust Detection Threshold",
    options=["Low", "Medium", "High"],
    value="Medium"
)

sensitivity_map = {"Low": 1.5, "Medium": 1.0, "High": 0.8}
multiplier = sensitivity_map[sensitivity]
st.sidebar.info(f"Threshold Multiplier: **{multiplier}x**")

# --- LOAD DATA ---
X_data, y_labels = load_sample_data()

if 'stream_index' not in st.session_state: st.session_state.stream_index = 0
if 'history' not in st.session_state: st.session_state.history = []

# --- DASHBOARD LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_normal = col2.empty()
metric_attack = col3.empty()
status_indicator = col4.empty()

chart_col, log_col = st.columns([2, 1])

with chart_col:
    st.markdown("#### Live Anomaly Score")
    chart_placeholder = st.empty()
    explain_placeholder = st.empty()

with log_col:
    st.markdown("#### Traffic Log")
    log_placeholder = st.empty()

# --- SIMULATION LOOP ---
if st.button("▶️ Start Live Simulation"):
    
    if "Isolation" in model_choice:
        model = load_iso_forest()
    else:
        model, scaler, base_threshold = load_autoencoder()
        effective_threshold = base_threshold * multiplier

    for i in range(50):
        # Random Sample
        idx = np.random.randint(0, len(X_data))
        row = X_data.iloc[[idx]]
        actual_label = y_labels.iloc[idx]

        # --- PREDICTION LOGIC ---
        if "Isolation" in model_choice:
            score = model.decision_function(row)[0]
            iso_cutoff = 0.0 + (0.05 if sensitivity == "High" else -0.05 if sensitivity == "Low" else 0.0)
            is_anomaly = score < iso_cutoff
            # explain_placeholder.empty()
            
        else: 
            row_scaled = scaler.transform(row)
            reconstruction = model.predict(row_scaled)
            loss = np.mean(np.power(row_scaled - reconstruction, 2))
            score = -loss 
            is_anomaly = loss > effective_threshold
            
            # --- EXPLAINABILITY ---
            if is_anomaly:
                features = X_data.columns
                error_vector = np.power(row_scaled - reconstruction, 2)[0]
                explanation_df = pd.DataFrame({
                    'Feature': features,
                    'Error': error_vector
                }).sort_values(by='Error', ascending=False).head(3)
                
                with explain_placeholder.container():
                    st.error("🚨 ANOMALY CAUSE (Top Factors):")
                    st.bar_chart(explanation_df.set_index('Feature'))
            # else:
            #    explain_placeholder.empty()

        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        
        st.session_state.history.append({
            "Packet ID": idx,
            "Type": actual_label,
            "Score": round(score, 4),
            "Status": status
        })
        
        df_hist = pd.DataFrame(st.session_state.history)
        total_anomalies = len(df_hist[df_hist['Status'] == "🚨 ANOMALY"])
        
        metric_total.metric("Total Packets", len(df_hist))
        metric_normal.metric("Normal", len(df_hist) - total_anomalies)
        metric_attack.metric("Threats", total_anomalies)
        
        if is_anomaly:
            status_indicator.error(f"THREAT DETECTED")
        else:
            status_indicator.success("SECURE")
            
        chart_placeholder.line_chart(df_hist['Score'].tail(50))
        log_placeholder.dataframe(df_hist[['Type', 'Status']].tail(8), height=300)
        time.sleep(0.05)

# --- MLOPS: HUMAN FEEDBACK LOOP ---
st.markdown("---")
st.subheader("📝 MLOPS: Human Feedback Loop")

with st.expander("Report False Alarm (Teach the Model)"):
    st.write("Did the model flag a normal packet as an Anomaly? Select it below.")
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        anomalies = history_df[history_df['Status'] == "🚨 ANOMALY"]
        
        if not anomalies.empty:
            selected_id = st.selectbox("Select Packet ID to Report:", anomalies['Packet ID'])
            if st.button("Mark as False Positive (Safe)"):
                row_data = X_data.iloc[[selected_id]]
                feedback_file = "data/feedback_loop.csv"
                header = not os.path.exists(feedback_file)
                row_data.to_csv(feedback_file, mode='a', header=header, index=False)
                st.success(f"Packet #{selected_id} saved to Retraining Pipeline!")
        else:
            st.info("No anomalies detected yet.")
