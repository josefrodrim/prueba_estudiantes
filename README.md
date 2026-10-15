# Telco Customer Churn – Clasificación de fuga de clientes

Proyecto de ML para predecir si un cliente de telecomunicaciones abandonará el servicio,
usando el dataset **Telco Customer Churn** (IBM, 7.043 clientes, 21 columnas, ~26,5 % de churn).

## Arquitectura

```
          ┌──────────────┐   download.py    ┌──────────────┐  make_dataset.py  ┌────────────────────┐
 IBM URL ─▶│  data/raw/   │ ───────────────▶ │   clean()    │ ────────────────▶ │  data/processed/   │
          └──────────────┘                  └──────────────┘                   └─────────┬──────────┘
                                                                                         │ train.py
                         ┌───────────────────── sklearn Pipeline ─────────────────────┐  ▼
                         │ FeatureEngineer → ColumnTransformer (scaler + one-hot) → Modelo │
                         └────────────────────────────────────────────────────────────┘
                                │  CV estratificada (5 folds, ROC-AUC) sobre LogReg / RF / XGBoost
                                ▼
                     models/churn_model.joblib  ──▶  evaluate.py → reports/metrics.json + figures/
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        predict.py (batch CSV)        api/app.py (FastAPI /predict)
```

El preprocesamiento vive **dentro** del Pipeline, así que el mismo artefacto sirve para
entrenamiento e inferencia (sin fuga de datos entre train/test ni desajustes en producción).

## Estructura

```
.
├── configs/config.yaml          # Única fuente de verdad: rutas, features, hiperparámetros
├── data/
│   ├── raw/                     # CSV original (no versionado)
│   ├── interim/                 # Datos intermedios
│   └── processed/               # Datos limpios (parquet)
├── models/                      # Modelos entrenados (.joblib)
├── notebooks/                   # EDA y experimentos
├── reports/                     # metrics.json, predicciones, figures/
├── src/churn/
│   ├── config.py                # Carga de config y rutas
│   ├── data/
│   │   ├── download.py          # Descarga del dataset
│   │   └── make_dataset.py      # Limpieza (TotalCharges, target 0/1)
│   ├── features/
│   │   └── build_features.py    # Variables derivadas + ColumnTransformer
│   ├── models/
│   │   ├── registry.py          # Fábrica de modelos candidatos
│   │   ├── train.py             # CV, selección del mejor y guardado
│   │   ├── evaluate.py          # Métricas y gráficos en test
│   │   └── predict.py           # Inferencia batch
│   └── api/app.py               # Servicio REST
├── tests/                       # pytest con datos sintéticos
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Uso

```bash
python3 -m venv .venv && source .venv/bin/activate
make install      # dependencias + paquete en modo editable
make data         # descarga y limpia el dataset
make train        # entrena, elige el mejor modelo y evalúa
make predict      # predicciones batch -> reports/predictions.csv
make api          # http://127.0.0.1:8000/docs
make test
```

## Variables derivadas

| Variable | Descripción |
|---|---|
| `avg_charge_per_month` | `TotalCharges / tenure` |
| `num_services` | Cantidad de servicios contratados |
| `is_long_contract` | 1 si el contrato es de 1 o 2 años |

## Desbalance de clases

Se usa `class_weight="balanced"` (LogReg, RF) y `scale_pos_weight` (XGBoost). El umbral
de decisión (`model.threshold`) se puede ajustar en el config según el costo de negocio:
en churn suele convenir priorizar **recall** (no dejar escapar clientes en riesgo).

## Próximos pasos sugeridos

- Ajuste de hiperparámetros (`GridSearchCV` / Optuna) y del umbral óptimo.
- Explicabilidad con SHAP.
- Tracking de experimentos con MLflow.
- Dockerfile para la API.
