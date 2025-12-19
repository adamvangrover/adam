.PHONY: install test lint security types check-all run clean docker-build docker-up setup-ops

install:
	pip install -r requirements.txt
	pip install -e .

setup-ops:
	bash ops/setup.sh

check-all: setup-ops
	python ops/run_checks.py

test:
	python ops/checks/check_tests.py

lint:
	python ops/checks/check_lint.py

security:
	python ops/checks/check_security.py
	python ops/security/audit_config.py

types:
	python ops/checks/check_types.py

pulse:
	python scripts/launch_system_pulse.py

verify-pulse:
	timeout 10s python scripts/launch_system_pulse.py || true

test-frontend:
	cd services/webapp/client && npm test -- --watchAll=false

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
