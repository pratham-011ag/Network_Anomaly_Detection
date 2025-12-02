#!/bin/bash

echo "=========================================="
echo "🛡️  WEEKLY SECURITY UPDATE PROTOCOL"
echo "=========================================="
echo "📅 Date:" $(date)

# 1. Activate Environment
source venv/bin/activate

# 2. Run the Python Retraining Script
python src/9_retrain_model.py

# 3. Log the completion
if [ $? -eq 0 ]; then
    echo "✅ Retraining Protocol Complete."
else
    echo "❌ Error during retraining."
fi

