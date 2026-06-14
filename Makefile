.PHONY: install smoke train eval lint clean

install:
	pip install -r requirements.txt
	pip install -e .

smoke:
	python scripts/make_synthetic_sample.py --out data/smoke --num-samples 16
	python scripts/train.py --config configs/tiny_smoke.yaml

train:
	python scripts/train.py --config configs/default.yaml

eval:
	python scripts/evaluate.py --config configs/default.yaml --split test

clean:
	rm -rf runs outputs __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
