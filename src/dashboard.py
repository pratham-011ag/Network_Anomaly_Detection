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

# --- RUN LOOP ---
if st.button("▶️ Start Live Simulation"):
    
    # Load Model based on selection
    if "Isolation" in model_choice:
        model = load_iso_forest()
    else:
        model, scaler, base_threshold = load_autoencoder()
        # Apply Sensitivity Logic
        effective_threshold = base_threshold * multiplier

    for i in range(50):
        # Random Sample
        idx = np.random.randint(0, len(X_data))
        row = X_data.iloc[[idx]]
        actual = y_labels.iloc[idx]

        # --- PREDICTION LOGIC ---
        if "Isolation" in model_choice:
            score = model.decision_function(row)[0]
            # Sensitivity Logic
            iso_cutoff = 0.0 + (0.05 if sensitivity == "High" else -0.05 if sensitivity == "Low" else 0.0)
            is_anomaly = score < iso_cutoff
            
        else: # Autoencoder
            row_scaled = scaler.transform(row)
            reconstruction = model.predict(row_scaled)
            loss = np.mean(np.power(row_scaled - reconstruction, 2))
            score = -loss 
            is_anomaly = loss > effective_threshold

        # Visualization
        status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
        
        # --- FIXED HISTORY APPEND ---
        st.session_state.history.append({
            "Packet ID": idx,           # <--- THIS IS THE KEY FIX
            "Type": actual, 
            "Score": round(score, 4), 
            "Status": status
        })
        
        # Metrics
        df_hist = pd.DataFrame(st.session_state.history)
        total_anomalies = len(df_hist[df_hist['Status'] == "🚨 ANOMALY"])
        
        metric_total.metric("Total Packets", len(df_hist))
        metric_normal.metric("Normal", len(df_hist) - total_anomalies)
        metric_attack.metric("Threats", total_anomalies)
        
        if is_anomaly:
            status_indicator.error(f"THREAT: {actual}")
        else:
            status_indicator.success("SECURE")
            
        # Update Charts
        chart_placeholder.line_chart(df_hist['Score'].tail(50))
        log_placeholder.dataframe(df_hist[['Type', 'Status']].tail(8), height=300)
        
        time.sleep(0.05)

# --- MLOPS: FEEDBACK LOOP ---
st.markdown("---")
st.subheader("📝 MLOPS: Human Feedback Loop")

with st.expander("Report False Alarm (Teach the Model)"):
    st.write("Did the model flag a normal packet as an Anomaly? Select it below to add it to the training set.")
    
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
                st.success(f"Packet #{selected_id} saved to Feedback Loop! The model will learn this pattern next retraining cycle.")
        else:
            st.info("No anomalies detected yet to report.")
    else:
        st.info("Run the simulation first to generate data.")
