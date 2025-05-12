# Use the official Python image
FROM python:3.12.9

# Définir le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copier tout le contenu du projet dans le conteneur
COPY app.py metrics.py utils.py /app/

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Exposer le port 8000
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]