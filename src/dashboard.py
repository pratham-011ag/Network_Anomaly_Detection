import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Network Anomaly Guard",
    page_icon="🛡️",
    layout="wide"
)

# --- CONSTANTS ---
MODEL_PATH = "models/kdd_isolation_forest_fixed.pkl"

# --- LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample_data():
    # Load a small chunk of KDD data for simulation
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame
    
    # Quick Preprocessing (Same as Day 6.5)
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    # Save raw labels for display, then drop for prediction
    labels = df['labels'].copy()
    
    # Encode
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # Drop labels for the model input
    X = df.drop('labels', axis=1)
    
    # Return a mix of data (Normal + Attacks)
    return X, labels

# --- MAIN APP ---
st.title("🛡️ Network Anomaly Detection System")
st.markdown("### Real-Time Traffic Analysis (KDD Cup 99 Replay)")

# Load
model = load_model()
X_data, y_labels = load_sample_data()

# Session State for Simulation
if 'stream_index' not in st.session_state:
    st.session_state.stream_index = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# Layout: Metrics on top
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_normal = col2.empty()
metric_attack = col3.empty()
status_indicator = col4.empty()

# Layout: Charts
chart_col, log_col = st.columns([2, 1])
with chart_col:
    st.markdown("#### Live Anomaly Score (Negative = Bad)")
    chart_placeholder = st.empty()

with log_col:
    st.markdown("#### Traffic Log")
    log_placeholder = st.empty()

# Simulation Button
start_btn = st.button("▶️ Start Live Simulation")

if start_btn:
    # Run loop
    for i in range(100): # Simulate 100 packets
        # Get random sample from dataset to mix things up
        idx = np.random.randint(0, len(X_data))
        row = X_data.iloc[[idx]]
        actual_label = y_labels.iloc[idx]
        
        # Predict
        pred = model.predict(row)[0] # 1=Normal, -1=Anomaly
        score = model.decision_function(row)[0]
        
        # Interpretation
        is_anomaly = (pred == -1)
        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        color = "red" if is_anomaly else "green"
        
        # Update History
        st.session_state.history.append({
            "Packet ID": idx,
            "Type": actual_label,
            "Score": round(score, 4),
            "Status": status
        })
        
        # Update Metrics
        df_hist = pd.DataFrame(st.session_state.history)
        total_packets = len(df_hist)
        total_anomalies = len(df_hist[df_hist['Status'] == "🚨 ANOMALY"])
        
        metric_total.metric("Total Packets", total_packets)
        metric_normal.metric("Normal Traffic", total_packets - total_anomalies)
        metric_attack.metric("Threats Detected", total_anomalies)
        
        if is_anomaly:
            status_indicator.error(f"THREAT DETECTED: {actual_label}")
        else:
            status_indicator.success("System Secure")

        # Update Chart (Rolling Window)
        chart_data = df_hist['Score'].tail(50)
        chart_placeholder.line_chart(chart_data)
        
        # Update Log (Show last 10)
        log_placeholder.dataframe(
            df_hist[['Type', 'Score', 'Status']].tail(10), 
            height=300
        )
        
        # Speed Control
        time.sleep(0.1)
