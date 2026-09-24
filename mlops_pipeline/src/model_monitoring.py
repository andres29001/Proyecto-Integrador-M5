"""
model_monitoring.py
---------------------
Avance #3: Monitoreo y detección de data drift.

Responsabilidades de este script:
    - Comparar la distribución de los datos de producción/nuevos contra la
      distribución de los datos con los que se entrenó el modelo (baseline
      del EDA en comprension_eda.ipynb).
    - Calcular métricas de drift por columna (ej: PSI - Population Stability
      Index, o usar una librería como evidently).
    - Generar un reporte/alerta cuando el drift supere un umbral definido.
    - Este script alimenta el dashboard de Streamlit (Avance #3).

Placeholder: define el esqueleto a completar en el Avance #3.
"""

import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """TODO (Avance 3): implementar el cálculo real de PSI para detectar
    desplazamiento de una variable numérica entre el dataset de entrenamiento
    (expected) y el dataset nuevo (actual)."""
    raise NotImplementedError


def check_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, threshold: float = 0.2) -> dict:
    """TODO (Avance 3): recorrer columnas numéricas relevantes y devolver
    un dict {columna: psi_value}, marcando cuáles superan el umbral de alerta."""
    raise NotImplementedError


if __name__ == "__main__":
    print("Módulo de monitoreo de drift — pendiente de implementación (Avance 3).")
