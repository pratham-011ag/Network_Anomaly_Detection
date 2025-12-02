#!/bin/bash

echo "🚀 Starting Network Anomaly Detection System..."

# 1. Activate Virtual Environment
source venv/bin/activate

# 2. Check if model exists, if not, train it
if [ ! -f "models/kdd_isolation_forest_fixed.pkl" ]; then
    echo "⚠️ Model not found! Training now (this may take a minute)..."
    python src/6_real_world_fixed.py
fi

# 3. Run the Dashboard
echo "📊 Launching Dashboard..."
streamlit run src/dashboard.py
