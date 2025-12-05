#!/bin/bash
echo "🚀 Lancement de Flask (Gunicorn)..."
gunicorn -b 0.0.0.0:5000 serveur:app --timeout 600 --workers 1 --threads 2 &


#!/bin/bash

# =============================
# Azure App Service Startup Script
# =============================

# Exit immediately if a command exits with a non-zero status
set -e

# Nom de l'environnement virtuel
VENV_DIR="/home/site/wwwroot/antenv"

# Chemin de l'application
APP_DIR="/home/site/wwwroot"

# Script ou fichier Streamlit à exécuter
STREAMLIT_APP="main.py"

# Port Azure pour le serveur
PORT="${PORT:-80}"

echo "Starting Azure App Service startup script..."

# Vérifier si l'environnement virtuel existe
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment at $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Creating at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r "$APP_DIR/requirements.txt"
fi

# Assurer que nous sommes dans le bon dossier
cd "$APP_DIR"

# Lancer Streamlit avec les paramètres Azure
echo "Launching Streamlit app..."
streamlit run "$STREAMLIT_APP" --server.port "$PORT" --server.address 0.0.0.0

# Fin du script

