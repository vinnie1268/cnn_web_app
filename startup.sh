#!/bin/bash

echo "🔧 Installing compatible NumPy..."
pip install "numpy<2"

echo "🔧 Installing PyTorch CPU..."
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu

echo "🚀 Starting Flask app..."
exec gunicorn app.app:app --bind 0.0.0.0:8000 --timeout 180
