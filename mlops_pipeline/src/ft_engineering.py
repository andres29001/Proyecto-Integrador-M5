"""
ft_engineering.py
------------------
Avance #2: Ingeniería de características.

Responsabilidades de este script:
    - Cargar el checkpoint validado (o el csv crudo) generado en Cargar_datos.ipynb.
    - Limpiar columnas con datos sucios (ej: tendencia_ingresos, puntaje_datacredito).
    - Definir e imputar valores nulos.
    - Encodear variables categóricas.
    - Generar el dataset final listo para modelar y guardarlo (ej: data/processed.parquet).

Este archivo se deja como placeholder ejecutable: hoy solo carga y valida shape,
para que el pipeline no rompa. Se completa en el Avance #2 (ver diapositiva
"Detalle de Avance #2" del PI).
"""

import json
from pathlib import Path

import pandas as pd


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """TODO (Avance 2): la columna tendencia_ingresos mezcla valores numéricos
    con las categorías esperadas (Creciente/Decreciente/Estable). Definir regla
    de limpieza (ej: tratar los valores numéricos como dato faltante o
    reconstruir la categoría a partir de otra fuente)."""
    valid = {"Creciente", "Decreciente", "Estable"}
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(valid), other=pd.NA
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO (Avance 2): imputación, encoding, nuevas variables derivadas."""
    df = clean_tendencia_ingresos(df)
    df["puntaje_datacredito"] = pd.to_numeric(df["puntaje_datacredito"], errors="coerce")
    return df


def main():
    config = load_config()
    raw_path = Path(config["raw_data_path"])
    df = pd.read_csv(raw_path)
    df = build_features(df)
    print(f"Features listas: {df.shape[0]} filas x {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
