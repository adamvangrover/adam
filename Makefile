.PHONY: install test lint run clean docker-build docker-up

install:
	pip install -e .

test:
	pytest tests/

test-frontend:
	cd services/webapp/client && npm test -- --watchAll=false

lint:
	flake8 core/ services/ scripts/
	# black --check core/ services/ scripts/

run:
	python scripts/run_adam.py

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

docker-build:
	docker-compose build

docker-up:
	docker-compose up

setup-dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install
