.PHONY: install data train predict api test all clean

install:
	pip install -r requirements.txt && pip install -e .

data:
	python -m churn.data.download
	python -m churn.data.make_dataset

train:
	python -m churn.models.train

predict:
	python -m churn.models.predict data/raw/telco_churn.csv -o reports/predictions.csv

api:
	uvicorn churn.api.app:app --reload

test:
	pytest -q

all: data train

clean:
	rm -rf models/*.joblib reports/*.json reports/*.csv reports/figures/*.png data/processed/*.parquet
