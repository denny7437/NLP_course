.PHONY: help install dev test clean proto dev-gateway dev-ui

help:
	@echo "Команды для разработки:"
	@echo "  install     - Установка зависимостей"
	@echo "  dev         - Запуск dev окружения"
	@echo "  test        - Запуск тестов"
	@echo "  proto       - Генерация gRPC кода"
	@echo "  dev-gateway - Запуск Gateway с hot reload"
	@echo "  dev-ui      - Запуск Web UI"
	@echo "  clean       - Очистка временных файлов"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	cd web-ui && npm install

dev:
	docker-compose -f docker-compose.dev.yml up --build

test:
	pytest tests/ -v
	mypy services/

proto:
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/filter.proto

dev-gateway:
	cd services/gateway && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-ui:
	cd web-ui && npm run dev

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	docker system prune -f 