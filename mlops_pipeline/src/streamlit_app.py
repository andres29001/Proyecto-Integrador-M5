"""
streamlit_app.py
------------------
Avance #3: Aplicación en Streamlit.

Dos secciones:
    1. Predicción: formulario para simular un cliente nuevo y ver si el
       modelo predice que pagará a tiempo.
    2. Monitoreo de drift: dashboard con el PSI de cada variable, comparando
       el dataset base (entrenamiento) contra un lote "nuevo" (por defecto,
       el test set, a modo de demostración del mecanismo).

Correr con:
    streamlit run streamlit_app.py
"""

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from ft_engineering import (
    CATEGORIC_NOMINAL_COLS,
    CATEGORIC_ORDINAL_COLS,
    NUMERIC_COLS,
    TENDENCIA_ORDER,
    build_features,
    load_config,
    load_raw_data,
    split_X_y,
)
from model_monitoring import check_drift

st.set_page_config(page_title="Riesgo Crediticio", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("../models/best_model.pkl")


@st.cache_data
def load_dataset():
    config = load_config()
    df = load_raw_data(config["raw_data_path"])
    df = build_features(df)
    return df, config


st.title("📊 Modelo de Riesgo Crediticio")

tab_prediccion, tab_monitoreo = st.tabs(["🔮 Predicción", "📈 Monitoreo de Drift"])

# ------------------------------------------------------------------
# TAB 1: Predicción
# ------------------------------------------------------------------
with tab_prediccion:
    st.subheader("Simular un cliente nuevo")

    try:
        model = load_model()
        df_full, config = load_dataset()
        X_full, _ = split_X_y(df_full, config["target_column"])

        col1, col2, col3 = st.columns(3)
        input_data = {}

        with col1:
            input_data["capital_prestado"] = st.number_input("Capital prestado", min_value=0.0, value=float(X_full["capital_prestado"].median()))
            input_data["plazo_meses"] = st.number_input("Plazo (meses)", min_value=1, value=int(X_full["plazo_meses"].median()))
            input_data["edad_cliente"] = st.number_input("Edad del cliente", min_value=18, value=int(X_full["edad_cliente"].median()))
            input_data["salario_cliente"] = st.number_input("Salario del cliente", min_value=0, value=int(X_full["salario_cliente"].median()))
            input_data["cuota_pactada"] = st.number_input("Cuota pactada", min_value=0, value=int(X_full["cuota_pactada"].median()))
            input_data["total_otros_prestamos"] = st.number_input("Total otros préstamos", min_value=0, value=int(X_full["total_otros_prestamos"].median()))

        with col2:
            input_data["tipo_laboral"] = st.selectbox("Tipo laboral", options=X_full["tipo_laboral"].unique())
            input_data["tipo_credito"] = st.selectbox("Tipo de crédito", options=sorted(X_full["tipo_credito"].unique()))
            input_data["tendencia_ingresos"] = st.selectbox("Tendencia de ingresos", options=TENDENCIA_ORDER)
            input_data["puntaje_datacredito"] = st.number_input("Puntaje Datacrédito", min_value=0, max_value=999, value=int(X_full["puntaje_datacredito"].median()))
            input_data["promedio_ingresos_datacredito"] = st.number_input("Promedio ingresos (Datacrédito)", min_value=0.0, value=float(X_full["promedio_ingresos_datacredito"].median()))
            input_data["huella_consulta"] = st.number_input("Huella de consulta", min_value=0, value=int(X_full["huella_consulta"].median()))

        with col3:
            input_data["cant_creditosvigentes"] = st.number_input("Créditos vigentes", min_value=0, value=int(X_full["cant_creditosvigentes"].median()))
            input_data["creditos_sectorFinanciero"] = st.number_input("Créditos sector financiero", min_value=0, value=int(X_full["creditos_sectorFinanciero"].median()))
            input_data["creditos_sectorCooperativo"] = st.number_input("Créditos sector cooperativo", min_value=0, value=int(X_full["creditos_sectorCooperativo"].median()))
            input_data["creditos_sectorReal"] = st.number_input("Créditos sector real", min_value=0, value=int(X_full["creditos_sectorReal"].median()))
            input_data["saldo_mora"] = st.number_input("Saldo en mora", min_value=0.0, value=0.0)
            input_data["saldo_total"] = st.number_input("Saldo total", min_value=0.0, value=float(X_full["saldo_total"].median()))
            input_data["saldo_principal"] = st.number_input("Saldo principal", min_value=0.0, value=float(X_full["saldo_principal"].median()))
            input_data["saldo_mora_codeudor"] = st.number_input("Saldo en mora (codeudor)", min_value=0.0, value=0.0)

        if st.button("Predecir", type="primary"):
            input_df = pd.DataFrame([input_data])
            # Reordenar columnas según lo que espera el preprocesador
            expected_cols = NUMERIC_COLS + CATEGORIC_NOMINAL_COLS + CATEGORIC_ORDINAL_COLS
            input_df = input_df[[c for c in expected_cols if c in input_df.columns]]

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            if pred == 1:
                st.success(f"✅ El modelo predice: **paga a tiempo** (probabilidad: {proba[1]:.1%})")
            else:
                st.error(f"⚠️ El modelo predice: **NO paga a tiempo** (probabilidad: {proba[0]:.1%})")

    except FileNotFoundError:
        st.warning("No se encontró el modelo entrenado. Corré primero `python model_training_evaluation.py`.")

# ------------------------------------------------------------------
# TAB 2: Monitoreo de drift
# ------------------------------------------------------------------
with tab_monitoreo:
    st.subheader("Detección de Data Drift (PSI)")
    st.caption(
        "Compara la distribución de los datos de entrenamiento (baseline) contra "
        "un lote nuevo. Por defecto, para fines demostrativos, se usa el 20% de "
        "test como 'lote nuevo'. En producción real esto se reemplaza por un lote "
        "genuino de solicitudes recientes."
    )

    try:
        df_full, config = load_dataset()
        X_full, _ = split_X_y(df_full, config["target_column"])

        test_size = st.slider("Tamaño del 'lote nuevo' simulado (%)", 5, 50, 20) / 100
        baseline, nuevo = train_test_split(X_full, test_size=test_size, random_state=config.get("random_seed", 42))

        drift_report = check_drift(baseline, nuevo)

        n_alerta = (drift_report["nivel"] == "ALERTA (drift significativo)").sum()
        n_vigilar = (drift_report["nivel"] == "Vigilar (drift moderado)").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Columnas OK", int((drift_report["nivel"] == "OK").sum()))
        c2.metric("Columnas a vigilar", int(n_vigilar))
        c3.metric("Columnas en alerta", int(n_alerta))

        st.bar_chart(drift_report.set_index("columna")["psi"])
        st.dataframe(drift_report, width="stretch")

    except FileNotFoundError:
        st.warning("No se encontró el dataset. Verificá que Base_de_datos.csv esté en la raíz del repo.")
