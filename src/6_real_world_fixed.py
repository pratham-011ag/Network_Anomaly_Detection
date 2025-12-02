import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import os

# --- NEW IMPORT: Decouples the data logic from the model logic ---
from data_loader import load_data

MODEL_PATH = "models/kdd_isolation_forest_fixed.pkl"
os.makedirs("models", exist_ok=True)

def run_fixed_test():
    # --- 1. THE REPLACEMENT: LOAD DATA ---
    print("⚙️  Calling Data Loader...")
    
    # This one line replaces the previous 20 lines of fetching/cleaning code
    df = load_data()
    
    # --- 2. THE MODEL TRAINING (Kept the same) ---
    print("⚖️  Splitting Data (Training on Normal Only)...")
    normal_df = df[df['is_anomaly'] == 0]
    
    # Train on 50% of the normal data
    X_train, _ = train_test_split(normal_df, test_size=0.5, random_state=42)
    X_train = X_train.drop('is_anomaly', axis=1)
    
    print(f"🧠 Training Isolation Forest on {len(X_train)} samples...")
    model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
    model.fit(X_train)
    
    joblib.dump(model, MODEL_PATH)
    print("✅ Model Saved!")

if __name__ == "__main__":
    run_fixed_test()
