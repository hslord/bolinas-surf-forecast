# Variables
PYTHON    = python3
STREAMLIT = streamlit
SRC_DIR   = src
APP_MAIN  = $(SRC_DIR)/run_surf_app.py
APP_UI    = $(SRC_DIR)/surf_app_streamlit.py

.PHONY: help run ui install lint format clean

help:
	@echo "Usage:"
	@echo "  make install - Install/upgrade dependencies"
	@echo "  make run     - Run the data fetch and processing pipeline"
	@echo "  make ui      - Launch the Streamlit interactive dashboard"
	@echo "  make lint    - Run Pylint with Bolinas custom settings"
	@echo "  make format  - Auto-format code using Black"
	@echo "  make clean   - Remove cache and temporary files"

install:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	$(PYTHON) $(APP_MAIN)

ui:
	@echo "Launching Bolinas Surf Dashboard..."
	PYTHONPATH=$(SRC_DIR) $(STREAMLIT) run $(APP_UI)

lint:
	pylint --init-hook="import sys; sys.path.append('$(SRC_DIR)')" --disable=R,C $(SRC_DIR)

format:
	black $(SRC_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "Cleanup complete."