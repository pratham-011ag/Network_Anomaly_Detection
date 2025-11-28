import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
# We save data one level up from 'src', so we use ../data
DATA_PATH = "data/network_traffic.csv"
RESULTS_PATH = "results/day1_data_distribution.png"
N_SAMPLES = 10000
ANOMALY_RATIO = 0.05  # 5% bad packets

# Ensure directories exist (handling relative paths)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

def generate_synthetic_data():
    print(f"🚀 Generating {N_SAMPLES} network packets...")

    # --- 1. Normal Traffic ---
    n_normal = int(N_SAMPLES * (1 - ANOMALY_RATIO))
    # Normal: ~500 bytes, steady time gaps
    normal_size = np.random.normal(loc=500, scale=100, size=n_normal)
    normal_time = np.random.exponential(scale=10, size=n_normal)
    normal_ports = np.random.choice([80, 443, 8080, 22, 53], size=n_normal)
    normal_labels = np.zeros(n_normal)

    # --- 2. Anomaly Traffic ---
    n_anomaly = int(N_SAMPLES * ANOMALY_RATIO)
    # Anomaly: Huge packets (exfiltration) or tiny gaps (DDoS)
    anomaly_size = np.random.normal(loc=2000, scale=400, size=n_anomaly)
    anomaly_time = np.random.exponential(scale=0.5, size=n_anomaly) # Fast!
    anomaly_ports = np.random.randint(10000, 65535, size=n_anomaly)
    anomaly_labels = np.ones(n_anomaly)

    # --- 3. Merge & Shuffle ---
    df = pd.DataFrame({
        'packet_size': np.concatenate([normal_size, anomaly_size]),
        'inter_arrival_time': np.concatenate([normal_time, anomaly_time]),
        'source_port': np.concatenate([normal_ports, anomaly_ports]),
        'is_anomaly': np.concatenate([normal_labels, anomaly_labels])
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean up negatives
    df['packet_size'] = df['packet_size'].clip(lower=0)
    df['inter_arrival_time'] = df['inter_arrival_time'].clip(lower=0)

    return df

def visualize_data(df):
    print("📊 Creating visualization...")
    plt.figure(figsize=(10, 6))
    
    # Plot normal vs anomaly
    sns.scatterplot(
        data=df, 
        x='packet_size', 
        y='inter_arrival_time', 
        hue='is_anomaly', 
        palette={0: 'blue', 1: 'red'},
        alpha=0.6
    )
    
    plt.title("Network Traffic Distribution")
    plt.xlabel("Packet Size (bytes)")
    plt.ylabel("Time Gap (ms)")
    
    plt.savefig(RESULTS_PATH)
    print(f"✅ Graph saved to {RESULTS_PATH}")
    # plt.show() # Commented out so it doesn't block the terminal if no UI is present

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"💾 Dataset saved to {DATA_PATH}")
    visualize_data(df)
