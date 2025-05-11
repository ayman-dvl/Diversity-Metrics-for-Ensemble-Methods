# Utiliser l'image Docker `fastapi-metrics-app.tar`

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