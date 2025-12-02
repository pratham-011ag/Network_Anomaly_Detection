import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
MODEL_PATH = "models/kdd_isolation_forest_fixed.pkl"
FEEDBACK_PATH = "data/feedback_loop.csv"

def retrain_pipeline():
    print(f"🔄 STARTING MLOPS PIPELINE...")
    
    # 1. Check if there is feedback to learn from
    if not os.path.exists(FEEDBACK_PATH):
        print("✅ No new feedback data found. Model is up to date.")
        return

    print("📈 New feedback data detected! Retraining model...")

    # 2. Load New Feedback Data
    feedback_df = pd.read_csv(FEEDBACK_PATH)
    print(f"   - Found {len(feedback_df)} corrected samples.")

    # 3. Load Original Data (To maintain baseline memory)
    # (In production, you'd load a saved 'baseline.csv', here we fetch KDD again)
    print("🌍 Loading Baseline Data...")
    kdd_data = fetch_kddcup99(percent10=True, as_frame=True)
    df = kdd_data.frame
    
    # Quick Preprocessing (Same as before)
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Filter Normal Only
    df['is_anomaly'] = df['labels'].apply(lambda x: 0 if x == 'normal.' else 1)
    normal_df = df[df['is_anomaly'] == 0].drop(['labels', 'is_anomaly'], axis=1)
    
    # 4. MERGE: Baseline + Feedback
    # This teaches the model: "The old normal stuff is good, AND this new stuff is also good."
    combined_df = pd.concat([normal_df, feedback_df])
    
    # 5. RETRAIN
    print(f"🧠 Retraining Isolation Forest on {len(combined_df)} samples...")
    # We use a 50% split to keep it fast
    X_train, _ = train_test_split(combined_df, train_size=0.5, random_state=42)
    
    model = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
    model.fit(X_train)
    
    # 6. SAVE & CLEANUP
    joblib.dump(model, MODEL_PATH)
    
    # Rename feedback file so we don't process it twice (Archive it)
    os.rename(FEEDBACK_PATH, "data/feedback_processed.csv")
    
    print("✅ Model Successfully Updated & Saved!")
    print("🚀 The Feedback Loop is complete.")

if __name__ == "__main__":
    retrain_pipeline()
