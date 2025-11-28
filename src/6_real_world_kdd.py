import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- CONFIGURATION ---
MODEL_PATH = "models/kdd_isolation_forest.pkl"

def run_real_world_test():
    print("🌍 DOWNLOADING REAL-WORLD DATA (KDD Cup 99)...")
    print("   (This might take a minute the first time...)")
    
    # Fetch 10% of the dataset (about 494,000 rows) - manageable for your laptop
    kdd_data = fetch_kddcup99(percent10=True, as_frame=True)
    
    df = kdd_data.frame
    print(f"✅ Data Loaded! Shape: {df.shape}")

    # --- 1. DATA CLEANING ---
    print("🧹 Cleaning and Encoding Data...")
    
    # The dataset comes as 'bytes', we need to decode strings
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Separate Labels (What kind of attack is it?)
    # 'normal.' is normal traffic. Everything else (smurf, neptune, satan) is an attack.
    y = df['labels'].apply(lambda x: 0 if x == 'normal.' else 1) # 0=Normal, 1=Attack
    X = df.drop('labels', axis=1)

    # Encode Text Columns (Protocol, Service, Flag) to Numbers
    # e.g., 'tcp' -> 1, 'udp' -> 2
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # --- 2. TRAIN/TEST SPLIT ---
    # We train on a mix, but Isolation Forest works best if we train mostly on normal
    # But for this test, let's just throw the data at it and see if it finds the 
    # attacks as "outliers" naturally.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"📊 Training on {len(X_train)} packets...")

    # --- 3. TRAIN MODEL ---
    print("🧠 Training Isolation Forest (Real Data)...")
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.2, # We assume about 20% of network traffic might be malicious in this dataset
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train)
    joblib.dump(model, MODEL_PATH)

    # --- 4. EVALUATE ---
    print("🔮 Predicting on Test Set...")
    preds = model.predict(X_test)
    
    # Convert -1 (Anomaly) to 1, and 1 (Normal) to 0
    y_pred = [1 if x == -1 else 0 for x in preds]

    print("\n" + "="*40)
    print("🌍 KDD CUP 99 - PERFORMANCE REPORT")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Pred: Normal', 'Pred: Attack'],
                yticklabels=['Actual: Normal', 'Actual: Attack'])
    plt.title('Real World Data (KDD99) Performance')
    plt.savefig("results/kdd_matrix.png")
    print("📈 Matrix saved to results/kdd_matrix.png")

if __name__ == "__main__":
    run_real_world_test()
