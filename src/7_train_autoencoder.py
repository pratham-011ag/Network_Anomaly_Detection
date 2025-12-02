import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- NEW IMPORT ---
from data_loader import load_data

# CONFIGURATION
MODEL_PATH = "models/autoencoder_sklearn.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.npy"

def train_autoencoder():
    # 1. LOAD DATA (Decoupled)
    print("⚙️  Calling Data Loader...")
    df = load_data()
    
    # 2. PREPARE DATA (Train on Normal Only)
    print("⚖️  Splitting Data...")
    normal_df = df[df['is_anomaly'] == 0].drop('is_anomaly', axis=1)
    
    # Scale Data (0 to 1) - Critical for Neural Networks
    scaler = MinMaxScaler()
    x_normal_scaled = scaler.fit_transform(normal_df)
    joblib.dump(scaler, SCALER_PATH)
    
    # Split
    x_train, x_test = train_test_split(x_normal_scaled, test_size=0.2, random_state=42)
    
    # 3. TRAIN MODEL
    print("🧠 TRAINING NEURAL NETWORK (Scikit-Learn)...")
    # Architecture: Input -> 14 -> 7 -> 14 -> Output
    autoencoder = MLPRegressor(hidden_layer_sizes=(14, 7, 14), 
                               activation='relu', solver='adam', 
                               max_iter=50, random_state=42, verbose=True)
    
    autoencoder.fit(x_train, x_train)
    
    # 4. CALCULATE THRESHOLD
    reconstructions = autoencoder.predict(x_test)
    mse = np.mean(np.power(x_test - reconstructions, 2), axis=1)
    threshold = np.mean(mse) + 2 * np.std(mse)
    
    # 5. SAVE
    joblib.dump(autoencoder, MODEL_PATH)
    np.save(THRESHOLD_PATH, threshold)
    print(f"✅ Autoencoder Saved! Threshold: {threshold:.4f}")

if __name__ == "__main__":
    train_autoencoder()
