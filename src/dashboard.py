import os
# --- MAC OS FIX: Prevent Threading Deadlock ---
os.environ['OMP_NUM_THREADS'] = '1'
# ----------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Network Anomaly Guard", page_icon="🛡️", layout="wide")

# --- PATHS ---
# Using the Scikit-Learn versions to prevent Mac crashing
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
    # Load Scikit-Learn MLPRegressor (Safe for Mac)
    model = joblib.load(AUTOENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = np.load(THRESHOLD_PATH)
    return model, scaler, threshold

@st.cache_data
def load_sample_data():
    # Load 10% data for smooth simulation
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame
    
    # Cleaning & Encoding
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

# --- MAIN APP UI ---
st.title("🛡️ Network Anomaly Detection System")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("⚙️ Control Panel")

# 1. Engine Selection
model_choice = st.sidebar.selectbox(
    "Select Detection Engine:",
    ("Isolation Forest (Statistical)", "Autoencoder (Deep Learning)")
)

# 2. Sensitivity Slider
st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Sensitivity Level")
sensitivity = st.sidebar.select_slider(
    "Adjust Detection Threshold",
    options=["Low", "Medium", "High"],
    value="Medium"
)

# Logic map: High Sensitivity = Lower Threshold (Catch more)
sensitivity_map = {"Low": 1.5, "Medium": 1.0, "High": 0.8}
multiplier = sensitivity_map[sensitivity]

st.sidebar.info(f"Threshold Multiplier: **{multiplier}x**")

# --- LOAD DATA ---
X_data, y_labels = load_sample_data()

# Initialize Session State
if 'stream_index' not in st.session_state: st.session_state.stream_index = 0
if 'history' not in st.session_state: st.session_state.history = []

# --- DASHBOARD LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_normal = col2.empty()
metric_attack = col3.empty()
status_indicator = col4.empty()

# Create two columns: Charts (Left) and Log (Right)
chart_col, log_col = st.columns([2, 1])

with chart_col:
    st.markdown("#### Live Anomaly Score")
    chart_placeholder = st.empty()   # For the Line Chart
    explain_placeholder = st.empty() # For the Bar Chart (Explainability)

with log_col:
    st.markdown("#### Traffic Log")
    log_placeholder = st.empty()

# --- SIMULATION LOOP ---
if st.button("▶️ Start Live Simulation"):
    
    # Load selected model
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
            # Isolation Forest Logic
            score = model.decision_function(row)[0]
            # Dynamic Sensitivity Cutoff
            iso_cutoff = 0.0 + (0.05 if sensitivity == "High" else -0.05 if sensitivity == "Low" else 0.0)
            is_anomaly = score < iso_cutoff
            
            # Clear explanation chart for IsoForest (it doesn't support reconstruction error)
            explain_placeholder.empty()
            
        else: 
            # Autoencoder Logic (Deep Learning)
            row_scaled = scaler.transform(row)
            reconstruction = model.predict(row_scaled)
            
            # MSE (Total Error)
            loss = np.mean(np.power(row_scaled - reconstruction, 2))
            score = -loss 
            
            # Check Threshold
            is_anomaly = loss > effective_threshold
            
            # --- EXPLAINABILITY (XAI) ---
            if is_anomaly:
                features = X_data.columns
                # Error per feature
                error_vector = np.power(row_scaled - reconstruction, 2)[0]
                
                explanation_df = pd.DataFrame({
                    'Feature': features,
                    'Error': error_vector
                }).sort_values(by='Error', ascending=False).head(3)
                
                with explain_placeholder.container():
                    st.error("🚨 ANOMALY CAUSE (Top Factors):")
                    st.bar_chart(explanation_df.set_index('Feature'))
            else:
                explain_placeholder.empty()

        # Update Status
        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        
        # Save to History
        st.session_state.history.append({
            "Packet ID": idx,
            "Type": actual_label,
            "Score": round(score, 4),
            "Status": status
        })
        
        # Update Metrics
        df_hist = pd.DataFrame(st.session_state.history)
        total_anomalies = len(df_hist[df_hist['Status'] == "🚨 ANOMALY"])
        
        metric_total.metric("Total Packets", len(df_hist))
        metric_normal.metric("Normal", len(df_hist) - total_anomalies)
        metric_attack.metric("Threats", total_anomalies)
        
        if is_anomaly:
            status_indicator.error(f"THREAT: {actual_label}")
        else:
            status_indicator.success("SECURE")
            
        # Update Charts
        chart_placeholder.line_chart(df_hist['Score'].tail(50))
        log_placeholder.dataframe(df_hist[['Type', 'Status']].tail(8), height=300)
        
        time.sleep(0.05)

# --- MLOPS: HUMAN FEEDBACK LOOP ---
st.markdown("---")
st.subheader("📝 MLOPS: Human Feedback Loop")

with st.expander("Report False Alarm (Teach the Model)"):
    st.write("Did the model flag a normal packet as an Anomaly? Select it below to add it to the training set.")
    
    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        # Filter for anomalies
        anomalies = history_df[history_df['Status'] == "🚨 ANOMALY"]
        
        if not anomalies.empty:
            selected_id = st.selectbox("Select Packet ID to Report:", anomalies['Packet ID'])
            
            if st.button("Mark as False Positive (Safe)"):
                # Fetch data
                row_data = X_data.iloc[[selected_id]]
                feedback_file = "data/feedback_loop.csv"
                
                # Save
                header = not os.path.exists(feedback_file)
                row_data.to_csv(feedback_file, mode='a', header=header, index=False)
                
                st.success(f"Packet #{selected_id} flagged as False Positive! Saved to retraining pipeline.")
        else:
            st.info("No anomalies detected yet.")
    else:
        st.info("Run the simulation first to generate data.")
