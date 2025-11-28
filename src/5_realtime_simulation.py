import pandas as pd
import numpy as np
import joblib
import time
import sys
import random

# --- CONFIGURATION ---
MODEL_PATH = "models/isolation_forest.pkl"
FEATURES = ['packet_size', 'inter_arrival_time', 'flow_rate']

# ANSI Colors for Terminal Output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def generate_live_packet():
    """Generates a single random packet (Normal or Anomaly)"""
    is_attack = random.random() < 0.10  # 10% chance of being an attack
    
    if is_attack:
        # Generate weird data (Attack)
        size = np.random.normal(loc=2000, scale=400)
        interval = np.random.exponential(scale=0.5)
        ptype = "DDOS/Exfil"
    else:
        # Generate normal data
        size = np.random.normal(loc=500, scale=100)
        interval = np.random.exponential(scale=10)
        ptype = "Normal"
        
    # Ensure positive physics
    size = max(0, size)
    interval = max(0, interval)
    
    return {
        'packet_size': size,
        'inter_arrival_time': interval,
        'packet_type_actual': ptype # Just for us to verify visually
    }

def start_sentinel():
    print(f"{Colors.BOLD}🛡️  INITIALIZING NETWORK SENTINEL...{Colors.RESET}")
    print("🔄 Loading Model...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded. Listening for traffic...\n")
    print(f"{'Packet ID':<10} {'Size (B)':<10} {'Gap (ms)':<10} {'Flow Rate':<10} {'Prediction':<15}")
    print("-" * 65)

    packet_id = 1
    
    try:
        while True:
            # 1. Simulate a packet arriving
            raw_packet = generate_live_packet()
            
            # 2. Engineer Feature (Flow Rate)
            # Must match exactly how we trained the model!
            flow_rate = raw_packet['packet_size'] / (raw_packet['inter_arrival_time'] + 1e-5)
            
            # 3. Prepare Data for Model
            # Create a DataFrame with the exact same column names
            input_data = pd.DataFrame([{
                'packet_size': raw_packet['packet_size'],
                'inter_arrival_time': raw_packet['inter_arrival_time'],
                'flow_rate': flow_rate
            }])
            
            # 4. Predict
            # Model returns: 1 (Normal) or -1 (Anomaly)
            prediction_code = model.predict(input_data)[0]
            
            # 5. Calculate Anomaly Score (Lower is more anomalous)
            score = model.decision_function(input_data)[0]

            # 6. Display Result
            if prediction_code == -1:
                # ANOMALY DETECTED 🚨
                status = f"{Colors.RED}🚨 BLOCKED ({score:.2f}){Colors.RESET}"
            else:
                # NORMAL TRAFFIC ✅
                status = f"{Colors.GREEN}✅ PASS    ({score:.2f}){Colors.RESET}"

            # Print row
            print(f"{packet_id:<10} {int(raw_packet['packet_size']):<10} {raw_packet['inter_arrival_time']:.2f}{'':<6} {flow_rate:.2f}{'':<6} {status}")
            
            packet_id += 1
            
            # Wait a bit to simulate real traffic (0.5 seconds)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BOLD}🛑 SENTINEL STOPPED BY USER.{Colors.RESET}")
        sys.exit(0)

if __name__ == "__main__":
    start_sentinel()
