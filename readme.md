# Modelo de Riesgo Crediticio — PIM5 Data Science

## Caso de negocio

Una empresa financiera necesita **anticipar el comportamiento de pago de nuevos
solicitantes de crédito**, para reducir el riesgo de mora antes de otorgar un
préstamo. Como Científico de Datos del equipo de Datos y Analítica, el objetivo
de este proyecto es construir un modelo de aprendizaje automático supervisado
que, a partir de la información socioeconómica y de historial crediticio de un
solicitante, prediga si pagará o no a tiempo (`Pago_atiempo`), y disponibilizar
ese modelo de forma reproducible y monitoreable (principios de MLOps).

**Dataset:** 10.763 solicitudes históricas de crédito, 23 columnas (datos
demográficos, laborales, de ingresos y de historial crediticio del buró
Datacrédito).

## Estructura del proyecto

```
mlops_pipeline/
└── src/
    ├── config.json                    # configuración del proyecto (nombre, target, seed)
    ├── Cargar_datos.ipynb             # carga y validación estructural de datos
    ├── comprension_eda.ipynb          # EDA: univariable, bivariable, multivariable
    ├── ft_engineering.py              # limpieza + feature engineering
    ├── model_training_evaluation.py   # entrenamiento y evaluación de modelos
    ├── model_monitoring.py            # monitoreo y detección de data drift (PSI)
    ├── streamlit_app.py               # app: predicción + dashboard de monitoreo
    └── model_deploy.py                # API con FastAPI (Avance 4)
mlops_pipeline/
├── reports/                           # tablas y gráficos generados por el modelado
└── models/                            # modelo serializado (no versionado en git)
Base_de_datos.csv
requirements.txt
.gitignore
readme.md
```

## Setup del entorno

Windows (script provisto):
```
set_up.bat
```
Esto crea el entorno virtual, instala `requirements.txt` y lo registra como kernel de Jupyter.

Manual (Linux/Mac):
```bash
python3 -m venv riesgo_crediticio-venv
source riesgo_crediticio-venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=riesgo_crediticio-venv --display-name="riesgo_crediticio-venv Python ETL"
```

## Cómo correr cada parte

Todos los scripts se corren parado en `mlops_pipeline/src/`:

```bash
cd mlops_pipeline/src

# 1. Feature engineering (validación rápida, standalone)
python ft_engineering.py

# 2. Entrenar y comparar modelos, guarda el mejor en ../models/best_model.pkl
python model_training_evaluation.py

# 3. Ver reporte de drift por consola (demo train vs test)
python model_monitoring.py

# 4. App interactiva (predicción + dashboard de drift)
streamlit run streamlit_app.py
```

## Ramas del proyecto

- `developer`: desarrollo activo del avance en curso.
- `certification`: integración de lo desarrollado, previo a producción.
- `master`: versión estable / producción.

## Estado actual

- [x] Avance #1 — Estructura del repo, entorno, EDA (V1.0.0 / V1.0.1)
- [x] Avance #2 — Feature engineering y modelado (V1.1.0 / V1.0.1)
- [x] Avance #3 — Monitoreo de drift, app Streamlit, README final
- [x] Avance #4 — API con FastAPI + Docker
- [ ] Extra credit — SonarCloud

---

## 1. Análisis exploratorio (EDA)

Detalle completo en `mlops_pipeline/src/comprension_eda.ipynb`. Principales hallazgos:

- **Variable objetivo muy desbalanceada**: ~95% `Pago_atiempo=1` vs ~5% `Pago_atiempo=0`.
  Esto condiciona toda la estrategia de modelado y evaluación posterior.
- **`tendencia_ingresos`** contiene valores sucios: números sueltos mezclados con
  las 3 categorías esperadas (Creciente/Decreciente/Estable), probablemente por
  un error de carga en el sistema origen.
- **Nulos relevantes** en `promedio_ingresos_datacredito` y `tendencia_ingresos`
  (~27% cada una).
- Variables de historial crediticio (créditos por sector, huella de consulta,
  puntaje Datacrédito) muestran relación con el target, y son la base de las
  features usadas en el modelo.

## 2. Ingeniería de características

Preprocesamiento con `ColumnTransformer` (`ft_engineering.py`), con tres ramas:

| Rama | Columnas | Transformación |
|---|---|---|
| numeric | capital_prestado, plazo_meses, edad_cliente, salario_cliente, total_otros_prestamos, cuota_pactada, puntaje_datacredito, cant_creditosvigentes, huella_consulta, saldos, créditos por sector, promedio_ingresos_datacredito | `SimpleImputer` (mediana) |
| categoric | tipo_laboral, tipo_credito | `SimpleImputer` (moda) + `OneHotEncoder` |
| categoric ordinal | tendencia_ingresos (orden: Decreciente < Estable < Creciente) | `SimpleImputer` (moda) + `OrdinalEncoder` |

### Hallazgo crítico: fuga de datos (data leakage)

La columna **`puntaje`** separaba casi perfectamente ambas clases: el rango de
valores de "no paga a tiempo" y el de "paga a tiempo" prácticamente no se
superponían. Al entrenar con esa columna, los modelos daban ~100% de accuracy,
lo cual es la señal clásica de que la columna se calcula **después** de observar
el comportamiento real de pago (probablemente un score interno de post-mora).
Se excluyó del modelo, porque no estaría disponible al evaluar a un cliente nuevo.

*Nota abierta para discusión:* `saldo_mora`, `saldo_total`, `saldo_principal` y
`cant_creditosvigentes` describen el estado del propio crédito ya otorgado. Su
correlación con el target es baja (<0.08), por lo que no generan fuga numérica
medible, pero conceptualmente tampoco existirían para un cliente que todavía no
tiene el crédito. Se mantuvieron en esta versión como señal de comportamiento
histórico disponible en el sistema, pero queda documentado como una decisión
de modelado discutible.

## 3. Modelado y evaluación

Split 80/20 estratificado por `Pago_atiempo`. Se compensó el desbalance de
clases con `class_weight="balanced"` (Regresión Logística, Random Forest) y
`scale_pos_weight` (XGBoost). Resultados sobre el set de test:

| Modelo | Precision | Recall | F1 | ROC-AUC | Recall (no paga a tiempo) |
|---|---|---|---|---|---|
| Regresión Logística | 0.96 | 0.75 | 0.84 | 0.58 | **0.39** |
| XGBoost | 0.96 | 0.97 | 0.96 | 0.64 | 0.13 |
| Random Forest | 0.95 | 1.00 | 0.98 | 0.65 | 0.00 |

**Modelo seleccionado: Regresión Logística.** Tiene el ROC-AUC más bajo de los
tres, pero es el único que logra detectar una parte relevante (39%) de los
clientes que no pagan a tiempo. Random Forest tiene el mejor ROC-AUC pero
**recall de 0% en la clase de interés real del negocio**: nunca predice un
incumplimiento, lo que lo hace inútil para el objetivo del proyecto pese a sus
métricas globales aparentemente altas. Esto ilustra por qué, con clases muy
desbalanceadas, elegir el "mejor" modelo solo por accuracy o ROC-AUC puede ser
un error de negocio: hay que mirar la métrica alineada al objetivo real.

Ver `mlops_pipeline/reports/` para la tabla completa, curvas ROC y matrices de
confusión, y `mlops_pipeline/models/best_model.pkl` para el modelo serializado
(no versionado en git, se regenera con `model_training_evaluation.py`).

## 4. Monitoreo de data drift

`model_monitoring.py` calcula el **PSI (Population Stability Index)** de cada
variable, comparando la distribución base (entrenamiento) contra un lote nuevo:

| PSI | Interpretación |
|---|---|
| < 0.10 | Sin drift relevante |
| 0.10 – 0.25 | Drift moderado, vigilar |
| ≥ 0.25 | Drift significativo, revisar/reentrenar el modelo |

Para variables numéricas se usan cuantiles del baseline como buckets; para
categóricas se compara directamente la proporción de cada categoría. Como
demostración (no hay datos de producción reales todavía), el script compara
train vs test, lo cual correctamente no muestra drift porque provienen de la
misma distribución. En producción, `new_df` se reemplazaría por un lote real
de solicitudes recientes.

## 5. Aplicación en Streamlit

`streamlit_app.py` tiene dos secciones:

- **Predicción**: formulario para simular un cliente nuevo y ver la predicción
  del modelo (paga / no paga a tiempo) junto con la probabilidad.
- **Monitoreo de Drift**: dashboard interactivo con el PSI de cada variable,
  gráfico de barras y tabla detallada con nivel de alerta.

## 6. API con FastAPI y Docker

`model_deploy.py` expone el modelo mediante una API REST:

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Healthcheck simple |
| `/health` | GET | Confirma si el modelo entrenado está disponible |
| `/predict` | POST | Recibe los datos de un cliente, devuelve predicción + probabilidades |
| `/docs` | GET | Documentación interactiva (Swagger), autogenerada por FastAPI |

**Correr localmente (sin Docker):**
```bash
cd mlops_pipeline/src
uvicorn model_deploy:app --reload --host 0.0.0.0 --port 8000
# abrir http://localhost:8000/docs
```

**Correr con Docker:**
```bash
docker build -t riesgo-crediticio-api .
docker run -p 8000:8000 riesgo-crediticio-api
# abrir http://localhost:8000/docs
```

El `Dockerfile` instala `requirements.txt`, copia el proyecto, entrena el
modelo dentro de la imagen (para no depender de un `.pkl` externo) y levanta
la API con `uvicorn`. El `.dockerignore` excluye entornos virtuales, notebooks
checkpoints y artefactos pesados para mantener la imagen liviana.

## 7. Extra credit: SonarCloud

Pendiente: configurar SonarCloud sobre el repositorio para validar calidad de
código, seguridad, cobertura de pruebas y estilo.
