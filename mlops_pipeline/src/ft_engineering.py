"""
ft_engineering.py
------------------
Avance #2: Ingeniería de características.

Construye el preprocesamiento del dataset de riesgo crediticio usando un
ColumnTransformer con tres ramas, según el esquema definido para el proyecto:

    ColumnTransformer
    ├── numeric        -> SimpleImputer (mediana)
    ├── categoric       -> SimpleImputer (moda) + OneHotEncoder
    └── categoric ordinales -> SimpleImputer (moda) + OrdinalEncoder

Este módulo se importa desde model_training_evaluation.py: NO entrena modelos,
solo prepara los datos.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# --- Definición de grupos de columnas -----------------------------------

NUMERIC_COLS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

CATEGORIC_NOMINAL_COLS = ["tipo_laboral", "tipo_credito"]

# tendencia_ingresos SÍ tiene un orden real de negocio: una tendencia
# decreciente es "peor" que estable, que a su vez es "peor" que creciente.
CATEGORIC_ORDINAL_COLS = ["tendencia_ingresos"]
TENDENCIA_ORDER = ["Decreciente", "Estable", "Creciente"]

DROP_COLS = ["fecha_prestamo"]  # fecha del préstamo: no es predictiva para un cliente nuevo

TARGET_COLUMN_DEFAULT = "Pago_atiempo"


# --- Utilidades de carga y limpieza --------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    return pd.read_csv(Path(raw_data_path))


def clean_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """La columna tendencia_ingresos llega con valores numéricos sueltos
    mezclados con las 3 categorías válidas (ver hallazgo del EDA). Cualquier
    valor que no sea una de las 3 categorías esperadas se trata como nulo,
    y el imputer se encarga de rellenarlo con la moda."""
    df = df.copy()
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(TENDENCIA_ORDER), other=pd.NA
    )
    return df


def clean_tipo_credito(df: pd.DataFrame) -> pd.DataFrame:
    """tipo_credito es un código categórico (4, 9, 10, ...), no una cantidad:
    lo pasamos a texto para que el ColumnTransformer lo trate como nominal."""
    df = df.copy()
    df["tipo_credito"] = df["tipo_credito"].astype(str)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_tendencia_ingresos(df)
    df = clean_tipo_credito(df)
    df["puntaje_datacredito"] = pd.to_numeric(df["puntaje_datacredito"], errors="coerce")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def split_X_y(df: pd.DataFrame, target: str = TARGET_COLUMN_DEFAULT):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# --- ColumnTransformer ----------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Arma el ColumnTransformer con las 3 ramas: numeric / categoric / categoric ordinales."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categoric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    categoric_ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[TENDENCIA_ORDER])),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categoric", categoric_pipeline, CATEGORIC_NOMINAL_COLS),
            ("categoric_ordinal", categoric_ordinal_pipeline, CATEGORIC_ORDINAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Punto de entrada usado por otros scripts: limpia el dataframe crudo
    y lo deja listo para pasarlo al ColumnTransformer (sin transformarlo
    todavía: el ColumnTransformer se ajusta dentro del Pipeline de modelado
    para evitar fuga de datos entre train y test)."""
    return clean_data(df)


def main():
    config = load_config()
    df = load_raw_data(config["raw_data_path"])
    df = build_features(df)
    print(f"Dataset limpio: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Numéricas: {len(NUMERIC_COLS)} | Categóricas nominales: {len(CATEGORIC_NOMINAL_COLS)} "
          f"| Categóricas ordinales: {len(CATEGORIC_ORDINAL_COLS)}")


if __name__ == "__main__":
    main()
