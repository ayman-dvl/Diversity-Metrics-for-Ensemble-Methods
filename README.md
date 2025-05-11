# FastAPI Ensemble Metrics API
## Option 1 : passer par github
Cette partie du projet expose une API REST construite avec **FastAPI** permettant de :
- Uploader un dataset (CSV ou Excel)
- Uploader un ou plusieurs modèles (.pkl ou .h5)
- Calculer des métriques de diversité entre modèles (Q-statistic, correlation, etc.)

##  Prérequis

- Avoir **Docker** installé : https://www.docker.com/products/docker-desktop

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


## Option 2 : Utiliser l'image Docker `fastapi-metrics-app.tar`

Ce guide explique comment exécuter l'API FastAPI reçue sous forme d'image Docker (`.tar`) sans avoir besoin du code source.

---

##  Prérequis

- Avoir **Docker** installé : https://www.docker.com/products/docker-desktop
- Avoir téléchargé le fichier `fastapi-metrics-app.tar` depuis le lien Google Drive ci joint :  

---


### 1.  Télécharger l'image Docker

- Télécharge le fichier `fastapi-metrics-app.tar` depuis le lien partagé

---

### 2.  Ouvrir un terminal

- Lance PowerShell, CMD ou Terminal 
- Va dans le dossier contenant le `.tar`. Exemple :

```bash
cd "C:\Users\TonNom\Downloads"
```

---

### 3.  Charger l'image Docker

```bash
docker load -i fastapi-metrics-app.tar
```

Tu dois voir un message du type :
```
Loaded image: fastapi-metrics-app:latest
```

---

### 4. Lancer le conteneur Docker

```bash
docker run -p 8000:8000 fastapi-metrics-app
```

 Ton API sera maintenant accessible à l'adresse :
```
http://127.0.0.1:8000
```

---

### 5.  Tester l’API dans un navigateur

Ouvre :
```
http://127.0.0.1:8000/docs
```
Tu pourras interagir avec l'API via Swagger UI.

---
