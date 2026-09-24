"""
model_monitoring.py
---------------------
Avance #3: Monitoreo y detección de data drift.

Compara la distribución de los datos "nuevos" (ej: un lote reciente de
solicitudes) contra la distribución de los datos con los que se entrenó el
modelo (el baseline), usando el PSI (Population Stability Index), la métrica
estándar de la industria para esto.

Regla general de interpretación del PSI (por columna):
    PSI < 0.10             -> sin drift relevante
    0.10 <= PSI < 0.25      -> drift moderado, vigilar
    PSI >= 0.25            -> drift significativo, revisar/reentrenar

Este script puede correrse solo (compara train vs test como demostración del
mecanismo) o importarse desde streamlit_app.py para mostrar el dashboard.

Uso:
    python model_monitoring.py
"""

import numpy as np
import pandas as pd

from ft_engineering import (
    CATEGORIC_NOMINAL_COLS,
    CATEGORIC_ORDINAL_COLS,
    NUMERIC_COLS,
    build_features,
    load_config,
    load_raw_data,
    split_X_y,
)

PSI_ALERT_THRESHOLD = 0.25
PSI_WARNING_THRESHOLD = 0.10


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calcula el PSI de una variable NUMÉRICA entre el dataset base (expected,
    ej: datos de entrenamiento) y el dataset nuevo (actual, ej: producción).

    Se arman `buckets` divididos por los cuantiles de `expected`, y se compara
    qué proporción de cada dataset cae en cada bucket.
    """
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        # La variable tiene muy poca variabilidad para bucketizar de forma útil
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(expected, bins=breakpoints).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=breakpoints).value_counts(sort=False)

    expected_pct = (expected_counts / len(expected)).replace(0, 1e-6)
    actual_pct = (actual_counts / len(actual)).replace(0, 1e-6)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Misma lógica que population_stability_index pero para variables
    CATEGÓRICAS: los 'buckets' son directamente las categorías observadas."""
    expected_pct = expected.value_counts(normalize=True)
    actual_pct = actual.value_counts(normalize=True)

    categories = set(expected_pct.index) | set(actual_pct.index)
    psi = 0.0
    for cat in categories:
        e = expected_pct.get(cat, 1e-6) or 1e-6
        a = actual_pct.get(cat, 1e-6) or 1e-6
        psi += (a - e) * np.log(a / e)
    return float(psi)


def check_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Recorre las columnas numéricas y categóricas del proyecto y devuelve
    un DataFrame con el PSI de cada una, marcando el nivel de alerta."""
    rows = []

    for col in NUMERIC_COLS:
        if col in baseline_df.columns and col in new_df.columns:
            psi = population_stability_index(baseline_df[col], new_df[col])
            rows.append({"columna": col, "tipo": "numerica", "psi": psi})

    for col in CATEGORIC_NOMINAL_COLS + CATEGORIC_ORDINAL_COLS:
        if col in baseline_df.columns and col in new_df.columns:
            psi = categorical_psi(baseline_df[col], new_df[col])
            rows.append({"columna": col, "tipo": "categorica", "psi": psi})

    result = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def nivel(psi):
        if psi >= PSI_ALERT_THRESHOLD:
            return "ALERTA (drift significativo)"
        elif psi >= PSI_WARNING_THRESHOLD:
            return "Vigilar (drift moderado)"
        return "OK"

    result["nivel"] = result["psi"].apply(nivel)
    return result


def main():
    """Demostración: usa el propio split train/test como 'baseline' vs 'nuevo lote'.
    En producción real, 'new_df' sería un lote reciente de solicitudes (ej: la
    última semana), no el test set del entrenamiento."""
    from sklearn.model_selection import train_test_split

    config = load_config()
    df = load_raw_data(config["raw_data_path"])
    df = build_features(df)
    X, _y = split_X_y(df, config["target_column"])

    baseline, nuevo = train_test_split(X, test_size=0.2, random_state=config.get("random_seed", 42))

    drift_report = check_drift(baseline, nuevo)
    print(drift_report.to_string(index=False))

    n_alertas = (drift_report["nivel"] != "OK").sum()
    print(f"\n{n_alertas} de {len(drift_report)} columnas muestran drift moderado o significativo.")


if __name__ == "__main__":
    main()
