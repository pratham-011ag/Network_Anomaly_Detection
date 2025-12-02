# 🛡️ Real-Time Zero-Trust Network Anomaly Detection System

> A Semi-Supervised Intrusion Detection System (IDS) engineered to detect Zero-Day attacks and encrypted threats using Autoencoders and MLOps pipelines.

![Project Status](https://img.shields.io/badge/Status-Active-green?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Ensemble-orange?style=flat-square)
![MLOps](https://img.shields.io/badge/MLOps-Feedback%20Loop-blueviolet?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)

## 💡 Why This Project is Different
Most student IDS projects rely on **Supervised Learning** (training on known attacks), which fails against new threats. 

This project implements a **Zero-Trust (Semi-Supervised)** architecture:
1.  **Zero-Day Detection:** Trains *only* on normal traffic patterns. Anything deviating from the baseline (unknown attacks) is flagged.
2.  **Encrypted Traffic Analysis:** Uses statistical flow metrics (Variance & Burstiness) to detect anomalies in **HTTPS/TLS** traffic without decryption.
3.  **Self-Healing (MLOps):** Features a "Human-in-the-Loop" feedback pipeline to correct False Positives and retrain the model automatically to handle Concept Drift.

---

## 📸 Real-Time Dashboard (SOC Interface)
The system includes a Streamlit-based Security Operations Center (SOC) dashboard with **Explainable AI (XAI)**.

![Dashboard Screenshot](results/dashboard.jpg)
![Dashboard Screenshot](results/dashboard_1.jpg)

---

## 🛠️ System Architecture

### 1. The Engines (Hybrid Ensemble)
* **Engine A: Isolation Forest:** Fast, statistical outlier detection for massive volumetric attacks (DDoS).
* **Engine B: Deep Autoencoder (MLPRegressor):** A Neural Network that learns the non-linear "Shape" of normal traffic. High reconstruction error indicates an anomaly.

### 2. The Features
| Metric | Purpose | Solves Drawback |
| :--- | :--- | :--- |
| **Flow Rate** | Bytes per millisecond | Volumetric DDoS detection |
| **Inter-arrival Time** | Jitter analysis | Botnet synchronization detection |
| **Size Variance** | Payload consistency | **Encrypted Traffic Analysis** (HTTPS) |

### 3. MLOps Pipeline
* **Feedback Loop:** Analysts can mark "False Positives" in the UI.
* **Retraining:** `retrain.sh` automates the merging of feedback data with the baseline and updates the model version.

---

## 📂 Project Structure
```text
Network_Anomaly_Detection/
├── data/                      # Data storage (Dataset Agnostic)
├── models/                    # Serialized Models (.pkl) & Scalers
├── results/                   # Evaluation Metrics & Charts
├── src/
│   ├── data_loader.py         # Centralized Data Abstraction Layer
│   ├── 6_real_world_fixed.py  # Isolation Forest Training Pipeline
│   ├── 7_train_autoencoder.py # Autoencoder Training Pipeline
│   ├── 8_live_sniffer.py      # Real-Time Wi-Fi Sniffer (Scapy)
│   ├── 9_retrain_model.py     # MLOps Retraining Logic
│   └── dashboard.py           # Streamlit Web App (UI)
├── run.sh                     # One-Click Startup Script
├── retrain.sh                 # Automated Retraining Script
└── requirements.txt           # Dependencies
````

-----

## 🚀 How to Run

### Option 1: The Automation Script (Recommended)

This script sets up the environment, checks for models (training them if missing), and launches the dashboard.

```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual Control

**1. Train the Models**

```bash
python src/6_real_world_fixed.py    # Train Statistical Model
python src/7_train_autoencoder.py   # Train Deep Learning Model
```

**2. Launch Dashboard**

```bash
streamlit run src/dashboard.py
```

**3. Run Live Sniffer (Requires Root)**
*Captures real traffic from your Wi-Fi card.*

```bash
sudo ./venv/bin/python src/8_live_sniffer.py
```

-----

## 📊 Performance & Validation

The system was validated on the **KDD Cup 99** and **CIC-IDS2017** datasets.

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **Precision** | **99%** | Minimized False Positives to reduce Alert Fatigue. |
| **Recall** | **99%** | Successfully detected hidden attacks (Smurf, Neptune). |
| **Latency** | **\<2ms** | Optimized inference using Scikit-Learn (Thread-safe on MacOS). |

-----

## 🔮 Roadmap & Future Improvements

  * **Containerization:** Full Docker support for cloud deployment.
  * **Big Data Scaling:** Migration from Pandas to **Apache Spark** for multi-terabyte datasets.
  * **Adversarial Training:** Injecting noise into training data to harden the Autoencoder against evasion attacks.

-----

## ✍️ Author

**Aryan Gupta**



