# Imagen para disponibilizar el modelo de riesgo crediticio mediante FastAPI.
# Build:  docker build -t riesgo-crediticio-api .
# Run:    docker run -p 8000:8000 riesgo-crediticio-api
# Docs:   http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero (aprovecha la cache de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY mlops_pipeline/ ./mlops_pipeline/
COPY Base_de_datos.csv .

WORKDIR /app/mlops_pipeline/src

# El modelo (best_model.pkl) no viaja en la imagen por defecto (ver .dockerignore).
# Se entrena una vez al construir la imagen para que la API tenga qué servir.
RUN python model_training_evaluation.py

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
