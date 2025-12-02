import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(source="cicids2017"): # <--- Changed default to modern
    """
    Centralized Data Loader.
    Capable of loading KDD99 (Legacy) or CIC-IDS2017 (Modern).
    """
    if source == "kdd99":
        # ... (Keep your old KDD code here if you want) ...
        pass

    elif source == "cicids2017":
        print(f"🌍 Loading CIC-IDS2017 (Modern) from data/modern_traffic.csv...")
        
        # 1. Load Data (Assumes you placed the CSV in data/)
        # Using a sample chunk because the full file is huge
        try:
            df = pd.read_csv("data/modern_traffic.csv")
        except FileNotFoundError:
            print("❌ Error: 'data/modern_traffic.csv' not found.")
            print("👉 Please download Wednesday-workingHours.pcap_ISCX.csv from Kaggle and rename it.")
            return None

        # 2. Cleanup Column Names (Strip spaces)
        df.columns = df.columns.str.strip()
        
        # 3. Handle Infinity and NaN (Common in this dataset)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # 4. Filter Features
        # We select ~20 most relevant features to avoid crashing RAM
        # These are features that exist in both legacy and modern flows
        selected_features = [
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Fwd Packet Length Max",
            "Fwd Packet Length Min", "Bwd Packet Length Max",
            "Bwd Packet Length Min", "Flow Bytes/s", "Flow Packets/s",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Bwd IAT Total", "Packet Length Mean",
            "Label" # <--- Vital!
        ]
        
        # Only keep available columns
        existing_cols = [c for c in selected_features if c in df.columns]
        df = df[existing_cols]

        # 5. Create Anomaly Label (0=Benign, 1=Attack)
        # In this dataset, normal traffic is called "BENIGN"
        print("🏷️  Encoding Labels...")
        df['is_anomaly'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df = df.drop('Label', axis=1)
        
        # 6. Sampling (To match your machine's power)
        # Use 20% of the data (approx 150k rows)
        df = df.sample(frac=0.2, random_state=42)

        return df
    
    return None
