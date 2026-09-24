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
- [ ] Avance #2 — Feature engineering y modelado (V1.1.0 / V1.0.1)
- [ ] Avance #3 — Monitoreo de drift, app Streamlit, README final
- [ ] Avance #4 — API con FastAPI + Docker
- [ ] Extra credit — SonarCloud

## Hallazgos clave del EDA (resumen, ver `comprension_eda.ipynb` para el detalle)

- Variable objetivo desbalanceada: ~95% `Pago_atiempo=1` vs ~5% `Pago_atiempo=0`.
- `tendencia_ingresos` contiene valores sucios (numéricos mezclados con categorías).
- `puntaje_datacredito` llega como texto y debe convertirse a numérico.
- Nulos relevantes en `promedio_ingresos_datacredito` (~27%) y `tendencia_ingresos` (~27%).
