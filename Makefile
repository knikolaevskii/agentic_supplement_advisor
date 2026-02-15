.PHONY: install dev test seed api ui

install:
	python -m venv .venv
	.venv/bin/pip install -U pip setuptools wheel
	.venv/bin/pip install -e ".[dev]"

dev: api

api:
	.venv/bin/uvicorn app.main:app --reload --port 8000 &
	.venv/bin/streamlit run ui/streamlit_app.py --server.port 8501

test:
	.venv/bin/python -m pytest tests/ -v

seed:
	.venv/bin/python -m scripts.seed_kb
