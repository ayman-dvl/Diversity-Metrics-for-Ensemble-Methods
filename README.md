# FastAPI Ensemble Metrics API

Cette partie du projet expose une API REST construite avec **FastAPI** permettant de :
- Uploader un dataset (CSV ou Excel)
- Uploader un ou plusieurs modèles (.pkl ou .h5)
- Calculer des métriques de diversité entre modèles (Q-statistic, correlation, etc.)

##  Lancer le projet avec Docker

### 1. Cloner le dépôt
```bash
git clone https://github.com/ouiame10/Diversity-Metrics-for-Ensemble-Methods
cd Diversity-Metrics-for-Ensemble-Methods
```

### 2. Construire l'image Docker
```bash
docker build -t fastapi-metrics-app .
```

### 3. Lancer le conteneur
```bash
docker run -d -p 8000:8000 fastapi-metrics-app
```

### 4. Accéder à l'API
- Documentation interactive : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check : [http://127.0.0.1:8000/healthcheck](http://127.0.0.1:8000/healthcheck)

##  Endpoints disponibles

| Méthode | Endpoint                        | Description                                 |
|---------|----------------------------------|---------------------------------------------|
| POST    | /upload_dataset/                | Upload d’un fichier CSV ou Excel            |
| POST    | /upload_model/                  | Upload d’un seul modèle `.pkl` ou `.h5`     |
| POST    | /upload_models/                 | Upload de plusieurs modèles en une fois     |
| GET     | /models/                        | Liste des modèles chargés                   |
| POST    | /pairwise_metrics/              | Calcul des métriques entre 2 modèles        |
| POST    | /pairwise_metrics_for_models/   | Calcul des métriques pour plusieurs modèles |
| DELETE  | /delete_model/{model_name}      | Supprimer un modèle                         |
| DELETE  | /delete_dataset/                | Supprimer le dataset                        |

