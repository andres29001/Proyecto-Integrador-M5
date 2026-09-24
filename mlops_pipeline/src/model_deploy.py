"""
model_deploy.py
-----------------
Avance #4: Disponibilizar el modelo mediante una API (FastAPI).

Responsabilidades de este script:
    - Cargar el modelo entrenado y serializado (de model_training_evaluation.py).
    - Exponer un endpoint (ej: POST /predict) que reciba las features de un
      cliente y devuelva la predicción de riesgo (Pago_atiempo) y su probabilidad.
    - Validar el payload de entrada con pydantic.
    - Este archivo, junto con requirements.txt, es lo que se empaqueta en la
      imagen Docker (ver Detalle de Avance #4).

Placeholder: define el esqueleto de la API para completar en el Avance #4.
Correr localmente con:
    uvicorn model_deploy:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API Riesgo Crediticio", version="0.1.0")


class ClientePayload(BaseModel):
    """TODO (Avance 4): reflejar aquí las features finales que espera el modelo
    (las mismas que produce ft_engineering.py)."""
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    puntaje: float


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API de riesgo crediticio activa"}


@app.post("/predict")
def predict(payload: ClientePayload):
    """TODO (Avance 4): cargar el modelo serializado y devolver la predicción real."""
    return {
        "input": payload.model_dump(),
        "prediccion": None,
        "probabilidad": None,
        "nota": "Pendiente: cargar modelo entrenado (Avance 2) y aplicar aquí.",
    }
