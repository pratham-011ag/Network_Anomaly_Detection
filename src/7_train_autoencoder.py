import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# CONFIGURATION
MODEL_PATH = "models/autoencoder_sklearn.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.npy"

def train_autoencoder():
    print("🌍 LOADING DATA (KDD Cup 99)...")
    kdd_data = fetch_kddcup99(percent10=True, as_frame=True)
    df = kdd_data.frame

    # Cleaning
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    df['is_anomaly'] = df['labels'].apply(lambda x: 0 if x == 'normal.' else 1)
    df = df.drop('labels', axis=1)
    
    # Train on Normal Only
    normal_df = df[df['is_anomaly'] == 0].drop('is_anomaly', axis=1)
    
    scaler = MinMaxScaler()
    x_normal_scaled = scaler.fit_transform(normal_df)
    joblib.dump(scaler, SCALER_PATH)
    
    x_train, x_test = train_test_split(x_normal_scaled, test_size=0.2, random_state=42)
    
    print("🧠 TRAINING NEURAL NETWORK (Scikit-Learn)...")
    # Architecture: Input -> 14 -> 7 -> 14 -> Output
    autoencoder = MLPRegressor(hidden_layer_sizes=(14, 7, 14), 
                               activation='relu', solver='adam', 
                               max_iter=50, random_state=42, verbose=True)
    
    autoencoder.fit(x_train, x_train)
    
    # Threshold
    reconstructions = autoencoder.predict(x_test)
    mse = np.mean(np.power(x_test - reconstructions, 2), axis=1)
    threshold = np.mean(mse) + 2 * np.std(mse)
    
    joblib.dump(autoencoder, MODEL_PATH)
    np.save(THRESHOLD_PATH, threshold)
    print("✅ Model Saved!")

if __name__ == "__main__":
    train_autoencoder()
