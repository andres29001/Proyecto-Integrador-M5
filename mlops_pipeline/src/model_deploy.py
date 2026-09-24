"""
model_deploy.py
-----------------
Avance #4: Disponibilizar el modelo mediante una API (FastAPI).

Carga el modelo entrenado (mlops_pipeline/models/best_model.pkl, generado por
model_training_evaluation.py) y expone:

    GET  /            -> healthcheck simple
    POST /predict      -> recibe los datos de un cliente y devuelve la predicción

Correr localmente con:
    uvicorn model_deploy:app --reload --host 0.0.0.0 --port 8000

Luego probar en http://localhost:8000/docs (Swagger autogenerado por FastAPI).
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ft_engineering import CATEGORIC_NOMINAL_COLS, CATEGORIC_ORDINAL_COLS, NUMERIC_COLS, TENDENCIA_ORDER

MODEL_PATH = Path("../models/best_model.pkl")

app = FastAPI(
    title="API Riesgo Crediticio",
    description="Predice si un cliente nuevo pagará su crédito a tiempo.",
    version="1.0.0",
)

_model = None  # se carga de forma perezosa (lazy) en el primer request


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No se encontró el modelo en {MODEL_PATH.resolve()}. "
                    "Corré primero: python model_training_evaluation.py"
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class ClientePayload(BaseModel):
    """Features que espera el modelo (ver ft_engineering.py: NUMERIC_COLS,
    CATEGORIC_NOMINAL_COLS, CATEGORIC_ORDINAL_COLS). No incluye 'puntaje'
    (excluida del modelo por fuga de datos, ver README) ni 'fecha_prestamo'
    (no predictiva)."""

    capital_prestado: float = Field(..., example=5000000)
    plazo_meses: int = Field(..., example=24)
    edad_cliente: int = Field(..., example=35)
    salario_cliente: float = Field(..., example=2500000)
    total_otros_prestamos: float = Field(..., example=0)
    cuota_pactada: float = Field(..., example=250000)
    puntaje_datacredito: float = Field(..., example=750)
    cant_creditosvigentes: int = Field(..., example=2)
    huella_consulta: int = Field(..., example=3)
    saldo_mora: float = Field(..., example=0)
    saldo_total: float = Field(..., example=1000000)
    saldo_principal: float = Field(..., example=900000)
    saldo_mora_codeudor: float = Field(..., example=0)
    creditos_sectorFinanciero: int = Field(..., example=1)
    creditos_sectorCooperativo: int = Field(..., example=0)
    creditos_sectorReal: int = Field(..., example=1)
    promedio_ingresos_datacredito: float = Field(..., example=2500000)
    tipo_laboral: str = Field(..., example="Empleado")
    tipo_credito: str = Field(..., example="9")
    tendencia_ingresos: str = Field(..., example="Estable")

    class Config:
        json_schema_extra = {
            "example": {
                "capital_prestado": 5000000,
                "plazo_meses": 24,
                "edad_cliente": 35,
                "salario_cliente": 2500000,
                "total_otros_prestamos": 0,
                "cuota_pactada": 250000,
                "puntaje_datacredito": 750,
                "cant_creditosvigentes": 2,
                "huella_consulta": 3,
                "saldo_mora": 0,
                "saldo_total": 1000000,
                "saldo_principal": 900000,
                "saldo_mora_codeudor": 0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 2500000,
                "tipo_laboral": "Empleado",
                "tipo_credito": "9",
                "tendencia_ingresos": "Estable",
            }
        }


class PrediccionResponse(BaseModel):
    prediccion: str
    paga_a_tiempo: bool
    probabilidad_paga_a_tiempo: float
    probabilidad_no_paga_a_tiempo: float


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API de riesgo crediticio activa"}


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {"status": "ok" if model_ready else "modelo no encontrado", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PrediccionResponse)
def predict(payload: ClientePayload):
    model = get_model()

    input_df = pd.DataFrame([payload.model_dump()])
    expected_cols = NUMERIC_COLS + CATEGORIC_NOMINAL_COLS + CATEGORIC_ORDINAL_COLS
    input_df = input_df[expected_cols]

    if payload.tendencia_ingresos not in TENDENCIA_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"tendencia_ingresos debe ser una de: {TENDENCIA_ORDER}",
        )

    pred = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]

    return PrediccionResponse(
        prediccion="Paga a tiempo" if pred == 1 else "No paga a tiempo",
        paga_a_tiempo=bool(pred),
        probabilidad_paga_a_tiempo=round(float(proba[1]), 4),
        probabilidad_no_paga_a_tiempo=round(float(proba[0]), 4),
    )
