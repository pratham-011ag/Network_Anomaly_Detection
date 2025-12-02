import time
import numpy as np
import pandas as pd
import joblib
import sys
from scapy.all import sniff, IP, TCP, UDP
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/kdd_isolation_forest_fixed.pkl" 

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# --- STATE MEMORY ---
# We need to remember the last 20 packets to calculate Variance
packet_window = [] 
WINDOW_SIZE = 20 

print(f"{Colors.BLUE}🛡️  INITIALIZING ENCRYPTED TRAFFIC ANALYZER...{Colors.RESET}")

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found. Run script 6 first.")
    sys.exit()

model = joblib.load(MODEL_PATH)
last_time = time.time()

def calculate_encrypted_features(window):
    """
    Extracts features that work even on HTTPS/TLS traffic.
    """
    sizes = [x['size'] for x in window]
    times = [x['time'] for x in window]
    
    # 1. Size Variance (Math.var)
    # Botnets send identical packets (Variance ~ 0)
    # Real humans send random sizes (Variance > 1000)
    size_variance = np.var(sizes)
    
    # 2. Burstiness (Packets per second in this window)
    duration = times[-1] - times[0]
    if duration == 0: duration = 0.001
    burst_rate = len(window) / duration
    
    return size_variance, burst_rate

def process_packet(packet):
    global last_time, packet_window
    
    if IP in packet:
        current_time = time.time()
        pkt_size = len(packet)
        
        # Add to Window
        packet_window.append({'size': pkt_size, 'time': current_time})
        if len(packet_window) > WINDOW_SIZE:
            packet_window.pop(0) # Keep window size fixed
            
        # Only analyze if we have a full window
        if len(packet_window) == WINDOW_SIZE:
            
            # --- ADVANCED FEATURE ENGINEERING ---
            var, burst = calculate_encrypted_features(packet_window)
            
            status = f"{Colors.GREEN}✅ PASS{Colors.RESET}"
            
            # LOGIC:
            # Low Variance (< 50) + High Speed (> 50 pkts/s) = Automated Attack (DDoS/Bot)
            if var < 50 and burst > 50:
                status = f"{Colors.RED}🚨 ENCRYPTED ATTACK (BOTNET){Colors.RESET}"
            
            # Massive Variance (> 100,000) = Video Streaming (YouTube/Netflix)
            elif var > 100000:
                status = f"{Colors.BLUE}📺 STREAMING (SAFE){Colors.RESET}"

            print(f"Size: {pkt_size}B | Var: {var:.0f} | Burst: {burst:.1f}/s | {status}")

def start_sniffing():
    print(f"📡 Analyzing Flow Metrics (Variance & Burstiness)...")
    # Store=0 prevents RAM overflow
    sniff(filter="ip", prn=process_packet, store=0)

if __name__ == "__main__":
    start_sniffing()
