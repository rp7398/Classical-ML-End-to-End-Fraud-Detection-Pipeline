# ===============================
# Fraud Detection Pipeline
# Cross-platform Makefile
# ===============================

SHELL := /bin/bash

OS := $(shell uname -s)

ifeq ($(OS),Linux)
	PYTHON := pyvenv/bin/python
	PIP := pyvenv/bin/pip
endif

ifeq ($(OS),Darwin)
	PYTHON := pyvenv/bin/python
	PIP := pyvenv/bin/pip
endif

ifeq ($(OS),Windows_NT)
	PYTHON := pyvenv/Scripts/python
	PIP := pyvenv/Scripts/pip
endif

DOCKER_COMPOSE := docker compose
INFRA_DIR := fraud_detection/infra
AIRFLOW_DIR := fraud_detection/airflow
NETWORK := infra-net
VENV := pyvenv

.DEFAULT_GOAL := help

help:
	@echo "make up      -> start full system"
	@echo "make down    -> stop full system"
	@echo "make install -> install Python deps"

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

network:
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)

infra-up: network
	cd $(INFRA_DIR) && $(DOCKER_COMPOSE) up -d

infra-down:
	cd $(INFRA_DIR) && $(DOCKER_COMPOSE) down

airflow-db:
	docker exec -it infra-postgres-1 psql -U mlflow -d mlflowdb \
		-c "CREATE DATABASE IF NOT EXISTS airflow;" \
		-c "CREATE USER IF NOT EXISTS airflow WITH PASSWORD 'airflow';" \
		-c "GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;"

airflow-up:
	cd $(AIRFLOW_DIR) && $(DOCKER_COMPOSE) up -d

airflow-down:
	cd $(AIRFLOW_DIR) && $(DOCKER_COMPOSE) down

up: install infra-up airflow-db airflow-up
	@echo "System ready."

down: airflow-down infra-down
