"""
model_training_evaluation.py
-----------------------------
Avance #2: Entrenamiento y evaluación de modelos supervisados.

Responsabilidades de este script:
    - Cargar el dataset ya transformado por ft_engineering.py.
    - Separar en train/test (considerar el desbalance de clases visto en el EDA:
      ~95% Pago_atiempo=1 vs ~5% Pago_atiempo=0).
    - Entrenar al menos 2-3 modelos candidatos (ej: LogisticRegression,
      RandomForestClassifier, XGBoost).
    - Evaluar con métricas apropiadas para clases desbalanceadas: recall, F1,
      ROC-AUC, precision-recall, no solo accuracy.
    - Seleccionar y serializar el mejor modelo (ej: joblib/pickle) para que
      model_deploy.py lo pueda cargar.

Placeholder: define el esqueleto y las funciones a completar en el Avance #2.
"""

from ft_engineering import load_config, build_features
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_models(X_train, y_train):
    """TODO (Avance 2): entrenar y devolver un dict {nombre_modelo: modelo_entrenado}."""
    raise NotImplementedError


def evaluate_models(models: dict, X_test, y_test):
    """TODO (Avance 2): calcular recall, F1, ROC-AUC por modelo y devolver
    el mejor según el criterio que definas (justificarlo en el README)."""
    raise NotImplementedError


def main():
    config = load_config()
    df = pd.read_csv(config["raw_data_path"])
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df, config["target_column"], seed=config["random_seed"])
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
