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
MODEL_PATH = "models/kdd_isolation_forest_fixed.pkl"

def run_fixed_test():
    print("🌍 LOADING KDD CUP 99 DATA...")
    kdd_data = fetch_kddcup99(percent10=True, as_frame=True)
    df = kdd_data.frame
    
    # --- 1. PREPROCESSING ---
    print("🧹 Cleaning Data...")
    
    # Decode bytes to strings
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Label Encoding (Text -> Numbers)
    # We must keep track of encoders to reverse them if needed, but for now we just transform
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Define Labels: 0 = Normal, 1 = Attack
    # The dataset uses 'normal.' (with a dot)
    df['is_anomaly'] = df['labels'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # Drop the original label text column
    df = df.drop('labels', axis=1)

    # --- 2. INTELLIGENT SPLIT ---
    print("⚖️  Splitting Data (Training on Normal Only)...")
    
    # Separate Normal and Attack
    normal_df = df[df['is_anomaly'] == 0]
    attack_df = df[df['is_anomaly'] == 1]
    
    # Training Set: 50% of the NORMAL data
    # We DO NOT show the model any attacks yet.
    X_train, X_test_normal = train_test_split(normal_df, test_size=0.5, random_state=42)
    
    # Test Set: Remaining 50% Normal + ALL Attacks
    test_df = pd.concat([X_test_normal, attack_df]).sample(frac=1, random_state=42)
    
    # Prepare Inputs (Drop label column)
    X_train_features = X_train.drop('is_anomaly', axis=1)
    X_test_features = test_df.drop('is_anomaly', axis=1)
    y_test_labels = test_df['is_anomaly']

    print(f"   - Train Size (Normal Only): {len(X_train_features)}")
    print(f"   - Test Size (Mixed):        {len(X_test_features)}")

    # --- 3. TRAIN MODEL ---
    print("🧠 Training Isolation Forest...")
    # contamination='auto': Model determines threshold based on decision function
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    
    model.fit(X_train_features)
    joblib.dump(model, MODEL_PATH)

    # --- 4. EVALUATE ---
    print("🔮 Predicting...")
    preds = model.predict(X_test_features)
    
    # Map: -1 (Anomaly) -> 1, 1 (Normal) -> 0
    y_pred = [1 if x == -1 else 0 for x in preds]

    print("\n" + "="*40)
    print("🏆 FIXED REAL-WORLD PERFORMANCE")
    print("="*40)
    print(classification_report(y_test_labels, y_pred, target_names=['Normal', 'Attack']))

    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Pred: Normal', 'Pred: Attack'],
                yticklabels=['Actual: Normal', 'Actual: Attack'])
    plt.title('Fixed KDD99 Performance (Trained on Normal)')
    plt.savefig("results/kdd_fixed_matrix.png")
    print("✅ Matrix saved to results/kdd_fixed_matrix.png")

if __name__ == "__main__":
    run_fixed_test()
