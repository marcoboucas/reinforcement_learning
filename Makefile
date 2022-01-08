lint:
	python -m pylint src
	python -m flake8 src
	python -m mypy src

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

install:
	pip install wheel
	pip install -r requirements.txt