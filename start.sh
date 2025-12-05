#!/bin/bash
echo "🚀 Lancement de Flask (Gunicorn)..."
gunicorn -b 0.0.0.0:5000 serveur:app --timeout 600 --workers 1 --threads 2 &

echo "🚀 Lancement de Streamlit..."
streamlit run main.py --server.port 80 --server.address 0.0.0.0

