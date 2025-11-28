import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_PATH = "data/network_traffic.csv"
TRAIN_PATH = "data/train_data.csv"
TEST_PATH = "data/test_data.csv"

def process_data():
    print("🔄 Loading raw data...")
    df = pd.read_csv(INPUT_PATH)

    # --- 1. Feature Engineering ---
    # We create a new feature: 'flow_rate' (Bytes per millisecond)
    # High flow rate usually indicates a burst/flood attack.
    
    # Avoid division by zero by adding a tiny epsilon (1e-5)
    df['flow_rate'] = df['packet_size'] / (df['inter_arrival_time'] + 1e-5)
    
    print("✅ Features engineered: Added 'flow_rate'")

    # --- 2. Data Splitting Strategy ---
    # Goal: Train on mostly NORMAL data. Test on MIXED data.
    
    # Separate Normal and Anomaly
    normal_df = df[df['is_anomaly'] == 0]
    anomaly_df = df[df['is_anomaly'] == 1]

    # Split Normal data: 80% for Training, 20% reserved for Testing
    train_normal, test_normal = train_test_split(normal_df, test_size=0.2, random_state=42)

    # CREATE TRAIN SET: Only Normal data (Unsupervised / One-Class approach)
    # The model will learn "This is what normal looks like."
    train_data = train_normal.copy()

    # CREATE TEST SET: The remaining Normal data + ALL Anomalies
    test_data = pd.concat([test_normal, anomaly_df]).sample(frac=1, random_state=42)

    # --- 3. Save Data ---
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print(f"\n💾 Data Saved:")
    print(f"   - Training Set ({len(train_data)} samples) -> {TRAIN_PATH}")
    print(f"   - Testing Set  ({len(test_data)} samples) -> {TEST_PATH}")
    
    print("\n🔍 Split Details:")
    print(f"   - Train Anomalies: {train_data['is_anomaly'].sum()} (Should be 0)")
    print(f"   - Test Anomalies:  {test_data['is_anomaly'].sum()} (All bad packets are here)")

if __name__ == "__main__":
    process_data()
