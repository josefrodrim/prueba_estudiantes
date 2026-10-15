# AGENTS.md

Guía para agentes de IA (Codex, Claude, Copilot, Cursor, etc.) que trabajen en este repositorio.
Léela completa antes de modificar código. Para la vista orientada a humanos, ver `README.md`.

---

## 1. Qué es este proyecto

Clasificación binaria de **fuga de clientes (churn)** de una empresa de telecomunicaciones.
Objetivo: predecir la probabilidad de que un cliente cancele el servicio (`Churn = 1`).

- **Contexto:** proyecto educativo (repositorio de estudiantes). Prioriza código claro,
  legible y bien estructurado por sobre trucos de optimización.
- **Idioma:** comentarios, docstrings, mensajes de consola y documentación en **español**.
  Los nombres de funciones/variables en Python van en inglés (`clean`, `build_preprocessor`).
- **Stack:** Python 3.10+ (probado en 3.14), pandas, scikit-learn, XGBoost, FastAPI, pytest.

## 2. Dataset

**Telco Customer Churn (IBM)**, descargado desde:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

- 7.043 filas, 21 columnas, una fila por cliente.
- Desbalanceado: **26,5 % churn** (1.869 positivos).

| Columna | Tipo crudo | Valores / notas |
|---|---|---|
| `customerID` | str | Identificador. **Nunca usar como feature.** |
| `gender` | str | `Male`, `Female` |
| `SeniorCitizen` | int | `0`/`1` → se convierte a `"No"`/`"Yes"` en `clean()` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | str | `Yes`, `No` |
| `MultipleLines` | str | `Yes`, `No`, `No phone service` |
| `InternetService` | str | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | str | `Yes`, `No`, `No internet service` |
| `Contract` | str | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | str | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `tenure` | int | Meses como cliente (0–72) |
| `MonthlyCharges` | float | Cargo mensual |
| `TotalCharges` | **str** | ⚠️ Viene como texto; 11 filas con `" "` (clientes con `tenure=0`). `clean()` lo convierte a float y rellena con 0. |
| `Churn` | str | **Target.** `Yes`/`No` → `1`/`0` en `clean()` |

Variables más predictivas conocidas: `Contract` (mes a mes ≫ churn), `tenure` (bajo ≫ churn),
`InternetService = Fiber optic`, `PaymentMethod = Electronic check`, ausencia de `TechSupport`/`OnlineSecurity`.

## 3. Arquitectura

```
IBM URL ─▶ data/raw/telco_churn.csv ─▶ clean() ─▶ data/processed/telco_churn_clean.parquet
                                                              │
                                                              ▼  train.py
         train_test_split estratificado (80/20, random_state=42)
                    │
                    ├─ X_train ─▶ StratifiedKFold(5) × cada candidato ─▶ ROC-AUC medio
                    │                    │
                    │                    ▼  se elige el mejor y se reentrena con todo X_train
                    │            models/churn_model.joblib  {"pipeline", "name", "threshold"}
                    │
                    └─ X_test ─▶ evaluate.py ─▶ reports/metrics.json, reports/figures/*.png

Inferencia:  predict.py (CSV batch)  ·  api/app.py (FastAPI POST /predict)
             ambos llaman a clean() + pipeline guardado
```

**Pipeline de scikit-learn (un único objeto serializado):**

```
Pipeline
├── preprocess: Pipeline
│   ├── engineer: FeatureEngineer          # crea variables derivadas (stateless)
│   └── columns: ColumnTransformer
│       ├── num: SimpleImputer(median) → StandardScaler
│       └── cat: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
└── model: LogisticRegression | RandomForestClassifier | XGBClassifier
```

### Principios de diseño (respetarlos)

1. **Todo el preprocesamiento va dentro del `Pipeline`.** Nada de `fit` sobre el dataset completo
   fuera del pipeline: provocaría *data leakage* en la validación cruzada.
2. **`clean()` solo hace transformaciones fila a fila sin estado** (tipos, mapeos). Lo que
   aprende de los datos (medianas, categorías, escalas) va en el pipeline.
3. **`configs/config.yaml` es la única fuente de verdad** para rutas, listas de features,
   hiperparámetros, umbral y semilla. No hardcodear estos valores en el código.
4. **Rutas siempre vía `churn.config.resolve()`**, relativas a la raíz del repo, nunca absolutas.
5. **El test set solo se usa una vez**, al final, en `evaluate()`. La selección de modelos
   se hace exclusivamente con CV sobre train.
6. **Semilla fija:** `project.random_state = 42` en todo lo aleatorio.

## 4. Mapa de archivos

| Archivo | Responsabilidad |
|---|---|
| `configs/config.yaml` | Configuración central |
| `src/churn/config.py` | `load_config()`, `resolve()`, `ROOT_DIR` |
| `src/churn/data/download.py` | Descarga el CSV a `data/raw/` (no re-descarga si existe) |
| `src/churn/data/make_dataset.py` | `clean(df)`, `load_raw()`; guarda parquet en `data/processed/` |
| `src/churn/features/build_features.py` | `FeatureEngineer`, `build_preprocessor(features_cfg)`, `SERVICE_COLS` |
| `src/churn/models/registry.py` | `get_model(name, params, random_state)` – fábrica de estimadores |
| `src/churn/models/train.py` | `split()`, CV de candidatos, selección, guardado, llama a `evaluate()` |
| `src/churn/models/evaluate.py` | Métricas (ROC-AUC, accuracy, precision, recall, F1), matriz de confusión, curva ROC |
| `src/churn/models/predict.py` | `predict(df)` → DataFrame con `churn_probability`, `churn_prediction`; CLI batch |
| `src/churn/api/app.py` | FastAPI: `GET /health`, `POST /predict` (esquema Pydantic `Customer`) |
| `tests/conftest.py` | Fixtures `cfg` y `sample_df` (40 filas sintéticas con el esquema real) |
| `tests/test_*.py` | Tests de limpieza, features y pipeline end-to-end |
| `notebooks/` | Exploración; convención `NN_autor_descripcion.ipynb` |

## 5. Comandos

Todos desde la raíz del repo con el entorno virtual activado.

```bash
python3 -m venv .venv && source .venv/bin/activate
make install   # pip install -r requirements.txt && pip install -e .
make data      # python -m churn.data.download && python -m churn.data.make_dataset
make train     # python -m churn.models.train
make predict   # python -m churn.models.predict data/raw/telco_churn.csv -o reports/predictions.csv
make api       # uvicorn churn.api.app:app --reload  → http://127.0.0.1:8000/docs
make test      # pytest -q
make clean     # borra modelos, métricas, figuras y parquet
```

- El paquete se instala en modo editable (`pip install -e .`, layout `src/`). Importar siempre como
  `from churn.<módulo> import ...`, nunca con rutas relativas ni `sys.path` hacks.
- `pytest` tiene `pythonpath = ["src"]` en `pyproject.toml`, funciona aunque no esté instalado.
- `make train` requiere haber corrido `make data`. `make predict`/`make api` requieren `make train`.

## 6. Validación y resultados actuales (línea base)

- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre el 80 % de train,
  métrica `roc_auc`, vía `cross_val_score`.
- **Desbalance:** `class_weight="balanced"` (LogReg, RF); `scale_pos_weight = neg/pos` calculado
  en `train.py` para XGBoost (no está en el YAML porque depende de los datos).
- **Umbral de decisión:** `model.threshold = 0.5` (configurable).

| Modelo | ROC-AUC CV (media ± std) |
|---|---|
| logistic_regression | 0.8456 ± 0.0124 |
| **random_forest** (seleccionado) | **0.8466 ± 0.0091** |
| xgboost | 0.8417 ± 0.0084 |

Test (1.409 filas): ROC-AUC 0.841 · accuracy 0.75 · clase Churn → precision 0.52, recall 0.78, F1 0.62.

Si un cambio empeora el ROC-AUC de CV respecto a esta línea base, menciónalo explícitamente.
En churn el negocio suele priorizar **recall** de la clase positiva.

### Limitaciones conocidas / trabajo pendiente

- No hay búsqueda de hiperparámetros (valores fijos del YAML). Si se agrega, usar CV anidada
  o al menos mantener el test set intacto.
- La CV solo reporta ROC-AUC; sería útil `cross_validate` con recall/precision/F1.
- Las diferencias entre modelos (~0.005) son menores que la std de CV: la selección no es concluyente.
- Sin optimización de umbral, sin SHAP, sin MLflow, sin Dockerfile.

## 7. Cómo hacer cambios comunes

### Agregar una variable derivada
1. Implementarla en `FeatureEngineer.transform()` (debe ser **sin estado**; si necesita aprender
   algo de los datos, crea un transformer con `fit` real).
2. Añadir su nombre a `features.engineered` en `config.yaml` (se trata como numérica).
3. Añadir un assert en `tests/test_features.py::test_feature_engineer_adds_columns`.

### Agregar/quitar una columna original
Editar `features.numeric` o `features.categorical` en `config.yaml`. Las columnas no listadas se
descartan (`remainder="drop"`). Si la columna es de entrada para la API, actualizar el modelo
Pydantic `Customer` en `api/app.py`.

### Agregar un modelo candidato
1. Añadir una rama en `get_model()` de `registry.py`.
2. Añadir su bloque con hiperparámetros en `model.candidates` del YAML.
3. Debe implementar `predict_proba` (se usa en evaluación e inferencia).

### Cambiar la métrica de selección
Cambiar `model.scoring` (cualquier string válido de scikit-learn, p. ej. `recall`, `f1`, `average_precision`).

## 8. Convenciones de código

- Estilo PEP 8, líneas ≤ 100 caracteres, type hints en firmas públicas.
- Cada módulo ejecutable expone `main()` y `if __name__ == "__main__": main()`; se ejecuta con `python -m`.
- `print()` para salida de CLI (no hay logging configurado); mensajes en español.
- Gráficos con backend `Agg` (sin GUI), guardados en `reports/figures/`.
- No introducir nuevas dependencias sin añadirlas a `requirements.txt`.
- Mantener las funciones pequeñas y el código didáctico; evitar abstracciones innecesarias.

## 9. Tests

- Ejecutar `make test` antes de dar un cambio por terminado; todos deben pasar.
- Los tests **no deben requerir descarga de datos ni modelo entrenado**: usar el fixture `sample_df`.
- `sample_df` incluye a propósito un `TotalCharges = " "` para cubrir el caso real de datos sucios.
- Todo cambio en `clean()`, `FeatureEngineer` o el pipeline debe venir con su test.

## 10. Qué NO hacer

- ❌ No versionar `data/`, `models/` ni `reports/` (ver `.gitignore`); solo los `.gitkeep`.
- ❌ No usar `customerID` como feature ni filtrarlo al modelo.
- ❌ No ajustar scalers/encoders/imputers fuera del pipeline ni sobre el dataset completo.
- ❌ No evaluar sobre el test set durante la selección de modelos o de hiperparámetros.
- ❌ No aplicar SMOTE/oversampling fuera del pipeline (si se usa, `imblearn.pipeline.Pipeline` dentro de la CV).
- ❌ No hardcodear rutas absolutas, hiperparámetros ni la semilla.
- ❌ No cambiar la firma de `predict(df)` ni el formato del bundle `.joblib`
  (`{"pipeline", "name", "threshold"}`) sin actualizar `predict.py` y la API.

## 11. Detalles del entorno

- `.venv/` local, ignorado por git. Python 3.14 funciona con las versiones de `requirements.txt`.
- `pyarrow` es necesario para leer/escribir parquet.
- `fastapi.testclient` emite un `StarletteDeprecationWarning` sobre `httpx`; es inofensivo.
- Repositorio remoto: https://github.com/josefrodrim/prueba_estudiantes (rama `main`).
