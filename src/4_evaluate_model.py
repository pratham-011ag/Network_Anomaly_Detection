import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
TEST_DATA_PATH = "data/test_data.csv"
MODEL_PATH = "models/isolation_forest.pkl"
RESULTS_PATH = "results/confusion_matrix.png"
FEATURES = ['packet_size', 'inter_arrival_time', 'flow_rate']

def evaluate_model():
    print("🔄 Loading model and test data...")
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_test = test_df[FEATURES]
    y_true = test_df['is_anomaly'] # The correct answers

    # --- PREDICTION ---
    print("🔮 Detecting anomalies...")
    # Model outputs: 1 (Normal), -1 (Anomaly)
    predictions = model.predict(X_test)
    
    # --- MAPPING OUTPUTS ---
    # We need to convert model outputs to match our labels (0 and 1)
    # Model:  1 -> 0 (Normal)
    # Model: -1 -> 1 (Anomaly)
    y_pred = [1 if x == -1 else 0 for x in predictions]

    # --- METRICS ---
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANCE REPORT")
    print("="*40)
    
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Check specific counts
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"✅ True Negatives (Normal traffic correctly identified): {tn}")
    print(f"🚨 True Positives (Attacks correctly detected):        {tp}")
    print(f"❌ False Positives (Normal flagged as attacks):        {fp}")
    print(f"⚠️ False Negatives (Attacks missed):                   {fn}")

    # --- VISUALIZATION ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Normal', 'Pred: Anomaly'],
                yticklabels=['Actual: Normal', 'Actual: Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(RESULTS_PATH)
    print(f"\n📈 Confusion Matrix saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()
