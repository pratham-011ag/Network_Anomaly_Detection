import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- CONFIGURATION ---
TRAIN_DATA_PATH = "data/train_data.csv"
MODEL_PATH = "models/isolation_forest.pkl"
FEATURES = ['packet_size', 'inter_arrival_time', 'flow_rate']

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

def train_model():
    print("🔄 Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Select only the features we want to train on
    # We DO NOT include 'is_anomaly' (That would be cheating!)
    X_train = train_df[FEATURES]
    
    print(f"🧠 Training Isolation Forest on {len(X_train)} records...")
    print(f"   - Features: {FEATURES}")

    # --- INITIALIZE MODEL ---
    # n_estimators=100: Create 100 trees (standard robust number)
    # contamination='auto': We let the model decide the threshold logic
    # random_state=42: Ensures reproducible results
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.01,  # We expect very few anomalies in "normal" training data
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # --- TRAIN (FIT) ---
    model.fit(X_train)
    print("✅ Model trained successfully!")

    # --- SAVE MODEL ---
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
