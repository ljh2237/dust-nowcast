.PHONY: download build train eval explain api web docker-up

download:
	python scripts/download_data.py --config configs/default.yaml

build:
	python scripts/build_dataset.py --config configs/default.yaml

train:
	python scripts/train.py --config configs/default.yaml --epochs 8

eval:
	python scripts/evaluate.py --config configs/default.yaml --model dustriskformer

explain:
	python scripts/explain.py --config configs/default.yaml

api:
	python scripts/run_api.py

web:
	python scripts/run_streamlit.py

docker-up:
	docker compose up --build
