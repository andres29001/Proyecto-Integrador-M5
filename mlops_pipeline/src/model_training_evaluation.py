"""
model_training_evaluation.py
-----------------------------
Avance #2: Entrenamiento y evaluación de modelos supervisados.

Flujo:
    1. Cargar y limpiar los datos (ft_engineering.py).
    2. Split 80/20 estratificado (el dataset está desbalanceado: ~95%/5%).
    3. Entrenar 3 modelos candidatos: Regresión Logística, Random Forest y
       XGBoost, todos compensando el desbalance de clases.
    4. Evaluar con métricas apropiadas para clases desbalanceadas (no solo
       accuracy): precision, recall, F1 y ROC-AUC.
    5. Generar una tabla comparativa y gráficos (matriz de confusión y curvas
       ROC superpuestas) y seleccionar el mejor modelo.
    6. Serializar el mejor modelo (joblib) para que model_deploy.py lo use.

Funciones reutilizables (pedidas explícitamente para evitar repetir código):
    - build_model(...): arma y entrena un Pipeline (preprocesador + clasificador).
    - summarize_classification(...): calcula y muestra las métricas de un modelo ya entrenado.

Uso:
    python model_training_evaluation.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ft_engineering import build_features, build_preprocessor, load_config, load_raw_data, split_X_y

RANDOM_SEED_DEFAULT = 42
REPORTS_DIR = Path("../reports")
MODELS_DIR = Path("../models")


# --- Función reutilizable: armar y entrenar un modelo ----------------------

def build_model(name: str, classifier, preprocessor, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Arma un Pipeline (preprocesador + clasificador), lo entrena y lo devuelve.
    Se usa igual para los 3 modelos, evitando repetir el mismo bloque de código."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)
    print(f"[{name}] entrenado sobre {X_train.shape[0]} registros.")
    return pipeline


# --- Función reutilizable: evaluar un modelo ya entrenado -------------------

def summarize_classification(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula precision, recall, F1 y ROC-AUC para un modelo ya entrenado,
    imprime el classification_report completo y devuelve un dict con el
    resumen (para armar la tabla comparativa entre modelos)."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo": name,
        # Métricas sobre la clase mayoritaria (Paga a tiempo = 1)
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        # Métrica crítica de negocio: ¿detecta a los que NO pagan a tiempo?
        # (clase minoritaria, 0). Un modelo con recall_no_paga = 0 es inútil
        # para el objetivo real del proyecto, aunque su ROC-AUC parezca bueno.
        "recall_no_paga": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["No paga a tiempo", "Paga a tiempo"], zero_division=0))
    print({k: round(v, 4) for k, v in metrics.items() if k != "modelo"})

    return metrics


# --- Gráficos comparativos --------------------------------------------------

def plot_confusion_matrices(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path):
    fig, axes = plt.subplots(1, len(pipelines), figsize=(5 * len(pipelines), 4))
    if len(pipelines) == 1:
        axes = [axes]
    for ax, (name, pipeline) in zip(axes, pipelines.items()):
        ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Matrices de confusión guardadas en: {output_path.resolve()}")


def plot_roc_curves(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipeline in pipelines.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_title("Curvas ROC comparativas")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Curvas ROC guardadas en: {output_path.resolve()}")


def plot_metrics_bar(comparison_df: pd.DataFrame, output_path: Path):
    metrics_cols = ["precision", "recall", "f1", "roc_auc", "recall_no_paga"]
    ax = comparison_df.set_index("modelo")[metrics_cols].plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_title("Comparación de métricas por modelo")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Gráfico comparativo de métricas guardado en: {output_path.resolve()}")


# --- Orquestación ------------------------------------------------------------

def main():
    config = load_config()
    target = config["target_column"]
    seed = config.get("random_seed", RANDOM_SEED_DEFAULT)

    df = load_raw_data(config["raw_data_path"])
    df = build_features(df)
    X, y = split_X_y(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas "
          f"(split 80/20 estratificado por {target})")

    # Compensación de desbalance de clases (~95% / ~5%)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    candidates = {
        "Regresion_Logistica": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Random_Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
        ),
    }

    preprocessor = build_preprocessor()

    pipelines = {}
    results = []
    for name, clf in candidates.items():
        pipeline = build_model(name, clf, preprocessor, X_train, y_train)
        metrics = summarize_classification(name, pipeline, X_test, y_test)
        pipelines[name] = pipeline
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    # Criterio de selección: priorizamos recall_no_paga (detectar clientes que
    # NO pagan a tiempo, el objetivo real de negocio) y usamos roc_auc como
    # desempate. Ordenar solo por roc_auc puede premiar un modelo que nunca
    # detecta al cliente moroso (recall_no_paga = 0), que es inútil en la práctica.
    comparison_df = comparison_df.sort_values(
        ["recall_no_paga", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    print("\n=== Tabla comparativa de modelos (ordenada por recall_no_paga, luego roc_auc) ===")
    print(comparison_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    plot_metrics_bar(comparison_df, REPORTS_DIR / "model_comparison_bar.png")
    plot_roc_curves(pipelines, X_test, y_test, REPORTS_DIR / "roc_curves.png")
    plot_confusion_matrices(pipelines, X_test, y_test, REPORTS_DIR / "confusion_matrices.png")

    best_model_name = comparison_df.iloc[0]["modelo"]
    best_pipeline = pipelines[best_model_name]
    print(f"\nMejor modelo según recall_no_paga (desempate por roc_auc): {best_model_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Modelo guardado en: {model_path.resolve()}")


if __name__ == "__main__":
    main()
