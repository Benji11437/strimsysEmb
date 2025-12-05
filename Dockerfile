# ===============================
# Étape 1 : Base Python optimisée
# ===============================
FROM python:3.10-slim

# Empêcher Python d’écrire des .pyc
ENV PYTHONDONTWRITEBYTECODE=1
# Afficher les logs en temps réel
ENV PYTHONUNBUFFERED=1

# ===============================
# Installation des dépendances système
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Créer un dossier pour l'application
# ===============================
WORKDIR /app

# ===============================
# Copier les dépendances Python
# ===============================
COPY requirements.txt .

# Installer les dépendances Python (TensorFlow inclus)
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copier le code dans l'image
# ===============================
COPY . .

# Donner permission au script
RUN chmod +x start.sh


# Exposer les ports
EXPOSE 8501    
EXPOSE 5000    

# ===============================
# Commande de lancement
# ===============================

CMD ["./start.sh"]

