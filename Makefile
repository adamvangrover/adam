.PHONY: install check test lint format clean docker-build docker-up setup-dev

install:
	pip install -r requirements.txt
	pip install -r ops/requirements.txt

check:
	python ops/run_checks.py

test:
	pytest tests/

lint:
	flake8 core/ services/
	mypy core/ services/

format:
	autopep8 --in-place --recursive core/ services/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	rm -f check_results.txt tests_output.txt
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

docker-build:
	docker-compose build

docker-up:
	docker-compose up

setup-dev:
	pip install -r requirements.txt
	pip install -e .
	bash ops/setup.sh
	pre-commit install