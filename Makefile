.PHONY: install dev test lint format demo advanced-demo dynamic-demo real-data-demo compare train-layer test-transplant clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

demo:
	./scripts/run_demo.sh

advanced-demo:
	./scripts/run_advanced_demo.sh

dynamic-demo:
	./scripts/run_dynamic_demo.sh

real-data-demo:
	./scripts/run_real_data_demo.sh examples/data_examples.jsonl

compare:
	python scripts/train_comparison.py --results results/tmu_vs_baseline.json

train-layer:
	python scripts/train_tmu_layer.py --fusion-layers 1 --export artifacts/tmu_layer.pt

test-transplant:
	python scripts/test_transplant.py --layer artifacts/tmu_layer.pt --results results/transplant_test.json

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
