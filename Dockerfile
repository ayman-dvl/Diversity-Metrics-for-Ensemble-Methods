# Utiliser une image de base Python légère
FROM python:3.9-slim

# Définir le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copier tout le contenu du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
