.PHONY: install dev test demo advanced-demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

demo:
	python -m tmu.cli --mode basic --epochs 1 --batch-size 2 --seq-len 32 --samples 64

advanced-demo:
	python -m tmu.cli --mode advanced --epochs 1 --batch-size 2 --seq-len 32 --samples 64 --max-gen-tokens 16

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
.PHONY: install dev test demo advanced-demo dynamic-demo real-data-demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

demo:
	./scripts/run_demo.sh

advanced-demo:
	./scripts/run_advanced_demo.sh

dynamic-demo:
	./scripts/run_dynamic_demo.sh

real-data-demo:
	./scripts/run_real_data_demo.sh examples/data_examples.jsonl

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +