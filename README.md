# Diversity Metrics For Ensemble Methods

# API Usage
## Option 1: Using GitHub  
This part of the project exposes a REST API built with **FastAPI** that allows you to:  
- Upload a dataset (CSV or Excel)  
- Upload one or more models (`.pkl`, `.keras` or `.h5`)  
- Calculate diversity metrics between models (Q-statistic, correlation, etc.)  

## Prerequisites  

- Have **Docker** installed: https://www.docker.com/products/docker-desktop  

## Launch the Project with Docker  

### 1. Clone the Repository  
```bash  
git clone https://github.com/ayman-dvl/Diversity-Metrics-for-Ensemble-Methods  
cd Diversity-Metrics-for-Ensemble-Methods  
```  

### 2. Build the Docker Image  
```bash  
docker build -t fastapi-metrics-app .  
```  

### 3. Run the Container  
```bash  
docker run -d -p 8000:8000 fastapi-metrics-app  
```  

### 4. Access the API  
- Interactive documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Health check: [http://127.0.0.1:8000/healthcheck](http://127.0.0.1:8000/healthcheck)  

## Available Endpoints  

| Method | Endpoint                        | Description                                 |  
|--------|----------------------------------|---------------------------------------------|  
| POST   | `/upload_dataset/`              | Upload a CSV or Excel file                 |  
| POST   | `/upload_model/`                | Upload a single (`.pkl`, `.keras` or `.h5`)  model      |  
| POST   | `/upload_models/`               | Upload multiple models at once             |  
| POST   | `/pairwise_metrics/`            | Calculate metrics between 2 models         |  
| POST   | `/pairwise_metrics_for_models/` | Calculate metrics for multiple models      |  
| GET    | `/models/`                      | Get list of uploaded models                         |  
| GET    | `/dataset/`                     | Get list of uploaded dataset   |
| DELETE | `/delete_model/{model_name}`    | Delete a model                             |  
| DELETE | `/delete_dataset/`              | Delete the dataset                         |  

## Option 2: Using the Docker Image `fastapi-metrics-app.tar`  

This guide explains how to run the FastAPI API received as a Docker image (`.tar`) without needing the source code.  

---  

## Prerequisites  

- Have **Docker** installed: https://www.docker.com/products/docker-desktop  

---  

### 1. Download the Docker Image  

- Download the `fastapi-metrics-app.tar` file from the shared link.  

---  

### 2. Open a Terminal  

- Launch PowerShell, CMD, or Terminal.  
- Navigate to the folder containing the `.tar` file. Example:  

### 3. Load the Docker Image  

```bash  
docker load -i fastapi-metrics-app.tar  
```  

You should see a message like:  
```  
Loaded image: fastapi-metrics-app:latest  
```  

---  

### 4. Run the Docker Container  

```bash  
docker run -p 8000:8000 fastapi-metrics-app  
```  

Your API will now be accessible at:  
```  
http://127.0.0.1:8000  
```  

---  

### 5. Test the API in a Browser  

Open:  
```  
http://127.0.0.1:8000/docs  
```  
You can interact with the API via Swagger UI.  

---  
