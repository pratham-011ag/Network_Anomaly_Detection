import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Centralized Data Loader.
    Currently fetches KDD99, but can be swapped for CIC-IDS2017
    without breaking the training scripts.
    """
    print("🌍 Data Loader: Fetching Dataset...")
    
    # 1. Load Data
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame
    
    # 2. Standard Cleaning (Decode Bytes)
    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # 3. Standard Encoding (Text -> Numbers)
    # Note: In a real production system, you would save these LabelEncoders 
    # to handle new data consistently.
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'labels':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # 4. Standard Labeling
    df['is_anomaly'] = df['labels'].apply(lambda x: 0 if x == 'normal.' else 1)
    df = df.drop('labels', axis=1)
    
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"✅ Data Loaded Successfully! Shape: {df.shape}")
