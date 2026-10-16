.PHONY: install run test all
install:
	pip install -r requirements.txt
run:
	python run_pipeline.py
test:
	pytest -q
all: install test run
