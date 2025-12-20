.PHONY: install check test lint format clean

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
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -f check_results.txt tests_output.txt
