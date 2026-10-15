# Notebooks

Convención de nombres: `NN_autor_descripcion.ipynb` (ej. `01_jr_eda.ipynb`).

- `01_eda` – análisis exploratorio (distribución del target, churn por contrato, tenure, cargos).
- `02_features` – prueba de nuevas variables antes de moverlas a `src/churn/features/`.
- `03_modelado` – experimentos rápidos; lo que funcione se lleva a `configs/config.yaml`.

Importar el paquete desde un notebook: `from churn.data.make_dataset import load_raw, clean`
(requiere `pip install -e .`).
