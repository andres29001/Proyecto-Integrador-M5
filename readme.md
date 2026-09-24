# Modelo de Riesgo Crediticio — PIM5 Data Science

Proyecto integrador de Henry: desarrollo y despliegue de un modelo de aprendizaje
automático para predecir si un nuevo cliente pagará a tiempo (`Pago_atiempo`) a
partir de su historial y datos socioeconómicos.

> Este README se completa a fondo en el **Avance #3** (caso de negocio, hallazgos y
> proceso). Por ahora documenta el setup y el estado del proyecto.

## Caso de negocio

Una empresa financiera necesita anticipar el comportamiento de pago de nuevos
solicitantes de crédito para reducir el riesgo de mora, usando su información
histórica de créditos.

## Estructura del proyecto

```
mlops_pipeline/
└── src/
    ├── config.json                    # configuración del proyecto (nombre, target, seed)
    ├── Cargar_datos.ipynb             # carga y validación estructural de datos
    ├── comprension_eda.ipynb          # EDA: univariable, bivariable, multivariable
    ├── ft_engineering.py              # limpieza + feature engineering (Avance 2)
    ├── model_training_evaluation.py   # entrenamiento y evaluación de modelos (Avance 2)
    ├── model_deploy.py                # API con FastAPI (Avance 4)
    └── model_monitoring.py            # monitoreo y data drift (Avance 3)
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

## Ramas del proyecto

- `developer`: desarrollo activo del avance en curso.
- `certification`: integración de lo desarrollado, previo a producción.
- `master`: versión estable / producción.

## Estado actual

- [x] Avance #1 — Estructura del repo, entorno, EDA (V1.0.0 / V1.0.1)
- [x] Avance #2 — Feature engineering y modelado (V1.1.0 / V1.0.1)
- [ ] Avance #3 — Monitoreo de drift, app Streamlit, README final
- [ ] Avance #4 — API con FastAPI + Docker
- [ ] Extra credit — SonarCloud

## Hallazgos clave del EDA (resumen, ver `comprension_eda.ipynb` para el detalle)

- Variable objetivo desbalanceada: ~95% `Pago_atiempo=1` vs ~5% `Pago_atiempo=0`.
- `tendencia_ingresos` contiene valores sucios (numéricos mezclados con categorías).
- `puntaje_datacredito` llega como texto y debe convertirse a numérico.
- Nulos relevantes en `promedio_ingresos_datacredito` (~27%) y `tendencia_ingresos` (~27%).

## Ingeniería de características (Avance #2)

Preprocesamiento con `ColumnTransformer` (ver `ft_engineering.py`):

| Rama | Columnas | Transformación |
|---|---|---|
| numeric | capital_prestado, plazo_meses, edad_cliente, salario_cliente, total_otros_prestamos, cuota_pactada, puntaje_datacredito, cant_creditosvigentes, huella_consulta, saldos, créditos por sector, promedio_ingresos_datacredito | `SimpleImputer` (mediana) |
| categoric | tipo_laboral, tipo_credito | `SimpleImputer` (moda) + `OneHotEncoder` |
| categoric ordinal | tendencia_ingresos (orden: Decreciente < Estable < Creciente) | `SimpleImputer` (moda) + `OrdinalEncoder` |

**Hallazgo crítico — fuga de datos (data leakage):** la columna `puntaje` separaba
perfectamente ambas clases (rango de "no paga a tiempo" y de "paga a tiempo" no se
superponen), lo que producía modelos con 100% de accuracy — señal inequívoca de que
esa columna se calcula *después* de observar el comportamiento de pago. Se excluyó
del modelo por no estar disponible al momento de evaluar a un cliente nuevo.

*Nota abierta:* `saldo_mora`, `saldo_total`, `saldo_principal` y `cant_creditosvigentes`
describen el estado del propio crédito ya otorgado. Su correlación con el target es baja
(<0.08), por lo que no generan fuga numérica, pero conceptualmente tampoco existirían
para un cliente que todavía no tiene el crédito. Se mantienen en esta versión como
señal de comportamiento histórico, documentado aquí para discusión.

## Modelado y evaluación (Avance #2)

Split 80/20 estratificado. Se compensó el desbalance de clases con `class_weight="balanced"`
(Regresión Logística, Random Forest) y `scale_pos_weight` (XGBoost). Resultados en test:

| Modelo | Precision | Recall | F1 | ROC-AUC | Recall (no paga a tiempo) |
|---|---|---|---|---|---|
| Regresión Logística | 0.96 | 0.75 | 0.84 | 0.58 | **0.39** |
| XGBoost | 0.96 | 0.97 | 0.96 | 0.64 | 0.13 |
| Random Forest | 0.95 | 1.00 | 0.98 | 0.65 | 0.00 |

**Modelo seleccionado: Regresión Logística.** Aunque tiene el ROC-AUC más bajo de los
tres, es el único que logra detectar una parte relevante (39%) de los clientes que no
pagan a tiempo. Random Forest tiene el mejor ROC-AUC pero **recall de 0% en la clase de
interés real del negocio**: nunca predice un incumplimiento, lo que lo hace inútil para
el objetivo del proyecto pese a sus métricas globales aparentemente altas. Esto ilustra
por qué, con clases muy desbalanceadas, elegir el "mejor" modelo solo por accuracy o
ROC-AUC puede ser un error — hay que mirar la métrica alineada al objetivo de negocio.

Ver `mlops_pipeline/reports/` para la tabla completa, curvas ROC y matrices de confusión,
y `mlops_pipeline/models/best_model.pkl` para el modelo serializado (no versionado en git).
