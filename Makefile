SRC_DIR := src

.PHONY: install lint format

install:
	pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --disable=R,C $(SRC_DIR)

format:
	black $(SRC_DIR)