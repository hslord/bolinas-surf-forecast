# Variables
PYTHON    = python
STREAMLIT = streamlit
SRC_DIR   = src
TEST_DIR  = test
SIM_DIR   = simulations
APP_MAIN  = $(SRC_DIR)/run_surf_app.py
APP_UI    = $(SRC_DIR)/surf_app_streamlit.py

.PHONY: help run ui install lint format test test-all coverage clean

help:
	@echo "Usage:"
	@echo "  make install  - Install/upgrade dependencies"
	@echo "  make run      - Run the data fetch and processing pipeline"
	@echo "  make ui       - Launch the Streamlit interactive dashboard"
	@echo "  make test     - Run unit tests (offline only)"
	@echo "  make test-all - Run all tests including slow network tests"
	@echo "  make coverage - Run tests with coverage report"
	@echo "  make lint     - Run Pylint with Bolinas custom settings"
	@echo "  make format   - Auto-format code using Black"
	@echo "  make clean    - Remove cache and temporary files"

install:
	pip install --upgrade pip && pip install -r requirements-pipeline.txt

run:
	$(PYTHON) $(APP_MAIN)

ui:
	@echo "Launching Bolinas Surf Dashboard..."
	PYTHONPATH=$(SRC_DIR) $(STREAMLIT) run $(APP_UI)

test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

coverage:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

lint:
	pylint --init-hook="import sys; sys.path.append('$(SRC_DIR)')" --disable=R,C $(SRC_DIR) $(TEST_DIR)

format:
	black $(SRC_DIR) $(TEST_DIR) $(SIM_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage
	@echo "Cleanup complete."
