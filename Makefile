# ==============================================================================
# MLOps & Data Engineering Orchestrator (Task 2: Pinecone) by IRONKAGE
# ==============================================================================

# 1. Експорт змінних середовища
ifneq (,$(wildcard ./.env))
	include .env
	export $(shell awk -F= '/^[a-zA-Z_]/ {print $$1}' .env)
endif

# --- Детектор рушія контейнерів (Docker або Podman) ---
ifneq (,$(shell command -v docker 2>/dev/null))
	DOCKER_CMD := docker
	COMPOSE_CMD := docker compose
else ifneq (,$(shell command -v podman 2>/dev/null))
	DOCKER_CMD := podman
	COMPOSE_CMD := podman compose
else
	DOCKER_CMD := none
endif

# 2. Кросплатформна підтримка ОС (Windows / Linux / MacOS) та Container Engine
ifeq ($(OS),Windows_NT)
	OPEN_CMD := start ""
	DOCKER_START_CMD := start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
	WAIT_DOCKER := powershell -Command "do { Write-Host '⏳ Чекаю на старт $(DOCKER_CMD)...'; Start-Sleep -Seconds 3 } while (!($(DOCKER_CMD) info 2>$$null))"
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OPEN_CMD := xdg-open
		DOCKER_START_CMD := systemctl --user start docker-desktop || sudo systemctl start docker
		WAIT_DOCKER := until $(DOCKER_CMD) info >/dev/null 2>&1; do echo "⏳ Чекаю на старт $(DOCKER_CMD)..."; sleep 3; done
	endif
	ifeq ($(UNAME_S),Darwin)
		OPEN_CMD := open
		DOCKER_START_CMD := open -a Docker
		WAIT_DOCKER := until $(DOCKER_CMD) info >/dev/null 2>&1; do echo "⏳ Чекаю на старт $(DOCKER_CMD)..."; sleep 3; done
	endif
endif

# 3. Змінні середовища та DRY версіонування
PY_VER := 3.12
# Магія GNU Make: автоматично видаляємо крапку (3.12 -> 312) для AUR
PY_VER_FLAT := $(subst .,,$(PY_VER))
PYTHON_CMD := python$(PY_VER)

VENV := venv
# Прапорець -u вимикає буферизацію. Логи та прогрес-бари виводитимуться МИТТЄВО!
PYTHON := $(VENV)/bin/python -u
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

# Кольори
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m
GRAY := \033[90m

# ------------------------------------------------------------------------------
# 🧠 SMART ROUTING & DYNAMIC HELP: Динамічний вибір бази та тексту
# ------------------------------------------------------------------------------
ifeq ($(strip $(ACTIVE_ENV)),cloud)
	ENV_LABEL := ☁️  Хмара (Pinecone SaaS Production)

	HELP_INFRA_UP   := $(GRAY)[Пропустити] Не потрібно для хмари (SaaS працює 24/7)$(RESET)
	HELP_INFRA_DOWN := $(GRAY)[Пропустити] Не потрібно для хмари$(RESET)
	HELP_DEEP_CLEAN := ПОВНЕ очищення (Лише Python кеші та локальні дані, $(DOCKER_CMD) не задіяний)
else
	ENV_LABEL := 🖥️  Локально ($(DOCKER_CMD) Pinecone Emulator)

	HELP_INFRA_UP   := Підняти локальний емулятор Pinecone у $(DOCKER_CMD)
	HELP_INFRA_DOWN := Зупинити контейнер емулятора Pinecone
	HELP_DEEP_CLEAN := ПОВНЕ очищення (Знищити локальні дані, датасети та зупинити $(DOCKER_CMD))
endif

.PHONY: help setup env ensure-python docker-ensure infra-up infra-down etl embed load search chunking hybrid dashboard clean deep-clean

help:
	@echo "$(CYAN)====================================================================================================$(RESET)"
	@echo "$(GREEN)🧠 ArXiv Vector Search Platform - Data Engineering Makefile | $(YELLOW)$(ENV_LABEL)$(RESET)"
	@echo "$(CYAN)====================================================================================================$(RESET)"
	@echo "Послідовність виконання проекту:"
	@echo "  $(YELLOW)[КРОК 0] Підготовка середовища:$(RESET)"
	@echo "    $(GREEN)make env$(RESET)           - Створити базовий .env файл (додайте ваші ключі Pinecone/Kaggle)"
	@echo "    $(GREEN)make setup$(RESET)         - Створити віртуальне середовище та встановити залежності"
	@echo "--------------------------------------------------------------------------------------------"
	@echo "  $(YELLOW)[КРОК 1] Інфраструктура бази даних (Vector DB):$(RESET)"
	@echo "    $(GREEN)make infra-up$(RESET)      - $(HELP_INFRA_UP)"
	@echo "--------------------------------------------------------------------------------------------"
	@echo "  $(YELLOW)[КРОК 2] Підготовка та завантаження (ETL):$(RESET)"
	@echo "    $(GREEN)make etl$(RESET)           - (01) Завантажити дані з Kaggle та конвертувати в Parquet"
	@echo "    $(GREEN)make embed$(RESET)         - (02) Згенерувати векторні ембеддинги (Specter2)"
	@echo "    $(GREEN)make load$(RESET)          - (03) Завантажити вектори та метадані в Pinecone"
	@echo "--------------------------------------------------------------------------------------------"
	@echo "  $(YELLOW)[КРОК 3] Пошук, Чанкінг та Аналітика:$(RESET)"
	@echo "    $(GREEN)make search$(RESET)        - (04) Семантичний пошук, фільтрація та порівняння метрик"
	@echo "    $(GREEN)make chunking$(RESET)      - (05) Порівняння Fixed-size vs Semantic chunking"
	@echo "    $(GREEN)make hybrid$(RESET)        - (06) Гібридний пошук (BM25 + Vector + Reciprocal Rank Fusion)"
	@echo "    $(GREEN)make dashboard$(RESET)     - Запустити інтерактивний BI-дашборд на Streamlit"
	@echo "--------------------------------------------------------------------------------------------"
	@echo "  $(YELLOW)[КРОК 4] Керування та очищення:$(RESET)"
	@echo "    $(GREEN)make infra-down$(RESET)    - $(HELP_INFRA_DOWN)"
	@echo "    $(GREEN)make clean$(RESET)         - Очистити кеші Python"
	@echo "    $(GREEN)make deep-clean$(RESET)    - $(HELP_DEEP_CLEAN)"
	@echo "$(CYAN)============================================================================================$(RESET)"

env:
	@if [ ! -f .env ]; then \
		echo "ACTIVE_ENV=local" > .env; \
		echo "KAGGLE_USERNAME=" >> .env; \
		echo "KAGGLE_KEY=" >> .env; \
		echo "PINECONE_API_KEY=your_actual_cloud_api_key_here" >> .env; \
		echo "PINECONE_LOCAL_HOST=http://127.0.0.1:5080" >> .env; \
		echo "$(GREEN)✅ Файл .env створено! Додайте ваші ключі доступу.$(RESET)"; \
	else \
		echo "$(YELLOW)⚡ Файл .env вже існує. Пропускаємо.$(RESET)"; \
	fi

# ------------------------------------------------------------------------------
# АВТОМАТИЗАЦІЯ PYTHON (Авто-встановлення та VENV)
# ------------------------------------------------------------------------------
ensure-python:
	@echo "$(CYAN)🔍 Перевірка наявності $(PYTHON_CMD)...$(RESET)"
	@command -v $(PYTHON_CMD) >/dev/null 2>&1 || { \
		echo "$(YELLOW)⚙️  $(PYTHON_CMD) не знайдено. Запускаю автоматичне встановлення...$(RESET)"; \
		if [ "$(OS)" = "Windows_NT" ] || [ -n "$$WINDIR" ]; then \
			echo "$(CYAN)🪟 Виявлено Windows. Встановлюю через PowerShell (winget)...$(RESET)"; \
			powershell -NoProfile -Command "winget install --id Python.Python.$(PY_VER) -e --silent --accept-package-agreements --accept-source-agreements"; \
		elif [ "$(UNAME_S)" = "Darwin" ]; then \
			echo "$(CYAN)🍏 Виявлено macOS. Встановлюю через Homebrew...$(RESET)"; \
			brew install python@$(PY_VER); \
		elif [ "$(UNAME_S)" = "Linux" ]; then \
			if command -v apt-get >/dev/null 2>&1; then \
				echo "$(CYAN)🟠 Виявлено Debian/Ubuntu. Встановлюю через APT...$(RESET)"; \
				sudo apt-get update && sudo apt-get install -y python$(PY_VER) python$(PY_VER)-venv; \
			elif command -v pacman >/dev/null 2>&1; then \
				echo "$(CYAN)👻 Виявлено Arch Linux. Шукаю специфічну версію Python $(PY_VER)...$(RESET)"; \
				if command -v yay >/dev/null 2>&1; then \
					echo "$(CYAN)📦 Знайдено AUR-хелпер 'yay'. Встановлюю python$(PY_VER_FLAT)...$(RESET)"; \
					yay -S --noconfirm python$(PY_VER_FLAT); \
				elif command -v paru >/dev/null 2>&1; then \
					echo "$(CYAN)📦 Знайдено AUR-хелпер 'paru'. Встановлюю python$(PY_VER_FLAT)...$(RESET)"; \
					paru -S --noconfirm python$(PY_VER_FLAT); \
				else \
					echo "$(YELLOW)❌ В офіційних репозиторіях Arch лише найновіший Python.$(RESET)"; \
					echo "$(YELLOW)👉 Для встановлення $(PY_VER) потрібен AUR. Виконайте вручну: yay -S python$(PY_VER_FLAT)$(RESET)" && exit 1; \
				fi; \
			elif command -v dnf >/dev/null 2>&1; then \
				echo "$(CYAN)🎩 Виявлено Fedora/RHEL. Встановлюю через DNF...$(RESET)"; \
				sudo dnf install -y python$(PY_VER); \
			else \
				echo "$(YELLOW)❌ Невідомий пакетний менеджер Linux. Встановіть Python $(PY_VER) вручну.$(RESET)" && exit 1; \
			fi; \
		else \
			echo "$(YELLOW)❌ Невідома ОС. Встановіть Python $(PY_VER) вручну з python.org$(RESET)" && exit 1; \
		fi; \
	}
	@echo "$(GREEN)✅ $(PYTHON_CMD) присутній у системі!$(RESET)"

setup: env ensure-python
	@echo "$(CYAN)📦 Створення віртуального середовища ($(PYTHON_CMD))...$(RESET)"
	$(PYTHON_CMD) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Віртуальне оточення готове!$(RESET)"

# ------------------------------------------------------------------------------
# АВТОМАТИЗАЦІЯ КОНТЕЙНЕРІВ (Перевірка, запуск та очікування)
# ------------------------------------------------------------------------------
docker-ensure:
	@echo "$(CYAN)[*] Перевірка наявності Container Engine (Docker/Podman)...$(RESET)"
	@if [ "$(DOCKER_CMD)" = "none" ]; then \
		echo "$(YELLOW)❌ Критична помилка: Docker або Podman не знайдено!$(RESET)\n👉 Встановіть Docker Desktop або Podman." && exit 1; \
	fi
	@echo "$(CYAN)[*] Знайдено рушій: $(DOCKER_CMD). Перевірка стану...$(RESET)"
	@$(DOCKER_CMD) info >/dev/null 2>&1 || (echo "$(YELLOW)[!] $(DOCKER_CMD) вимкнено. Виконую автоматичний запуск...$(RESET)" && $(DOCKER_START_CMD) && $(WAIT_DOCKER))
	@echo "$(GREEN)[+] $(DOCKER_CMD) готовий до роботи!$(RESET)"

infra-up:
	@if [ "$(strip $(ACTIVE_ENV))" = "cloud" ]; then \
		echo "$(YELLOW)⚡ Активне середовище - хмара (Pinecone SaaS). $(DOCKER_CMD) інфраструктура не потрібна.$(RESET)"; \
	else \
		$(MAKE) docker-ensure; \
		echo "$(CYAN)🐳 Запуск інфраструктури (Pinecone Local Emulator) через $(COMPOSE_CMD)...$(RESET)"; \
		$(COMPOSE_CMD) up -d; \
		echo "$(GREEN)✅ Локальний Pinecone доступний на 127.0.0.1:5080!$(RESET)"; \
	fi

infra-down:
	@if [ "$(strip $(ACTIVE_ENV))" = "cloud" ]; then \
		echo "$(YELLOW)⚡ Активне середовище - хмара (Pinecone SaaS). Інфраструктура не запущена.$(RESET)"; \
	else \
		$(MAKE) docker-ensure; \
		echo "$(YELLOW)🛑 Зупинка інфраструктури...$(RESET)"; \
		$(COMPOSE_CMD) down; \
		echo "$(GREEN)✅ Контейнер Pinecone зупинено.$(RESET)"; \
	fi

# ------------------------------------------------------------------------------
# ПАЙПЛАЙН ДАНИХ ТА АНАЛІТИКА
# ------------------------------------------------------------------------------
etl:
	@echo "$(CYAN)⏳ (01) Запуск ETL-ядра (Завантаження даних та конвертація у Parquet)...$(RESET)"
	$(PYTHON) scripts/01_prepare_data.py

embed:
	@echo "$(CYAN)🧠 (02) Генерація ембеддингів через Specter2...$(RESET)"
	$(PYTHON) scripts/02_embed.py

load:
	@echo "$(CYAN)☁️  (03) Завантаження векторів у Pinecone ($(ENV_LABEL))...$(RESET)"
	$(PYTHON) scripts/03_load_to_pinecone.py

search:
	@echo "$(CYAN)🔍 (04) Запуск семантичного пошуку та аналізу індексів...$(RESET)"
	$(PYTHON) scripts/04_search.py

chunking:
	@echo "$(CYAN)✂️ (05) Запуск тестування стратегій чанкінгу...$(RESET)"
	$(PYTHON) scripts/05_chunking.py

hybrid:
	@echo "$(CYAN)⚖️ (06) Запуск гібридного пошуку (BM25 + Vector + RRF)...$(RESET)"
	$(PYTHON) scripts/06_hybrid_search.py

dashboard:
	@echo "$(CYAN)📈 Запуск Streamlit Dashboard...$(RESET)"
	$(STREAMLIT) run dashboard/app.py

clean:
	@echo "$(YELLOW)🧹 Очищення тимчасових файлів...$(RESET)"
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Проект очищено!$(RESET)"

# Хардкорне знищення всього
deep-clean: clean
	@if [ "$(strip $(ACTIVE_ENV))" = "cloud" ]; then \
		echo "$(YELLOW)⚠️  ПОВНЕ ОЧИЩЕННЯ: Знищено локальні кеші та сирі дані. Хмарна БД не зачеплена.$(RESET)"; \
		rm -rf data/ embeddings/ arxiv-metadata-oai-snapshot.json arxiv.zip; \
	else \
		echo "$(YELLOW)⚠️  ПОВНЕ ОЧИЩЕННЯ: Видалення Томів, Датасетів та зупинка контейнерів...$(RESET)"; \
		$(COMPOSE_CMD) down -v || true; \
		rm -rf data/ embeddings/ arxiv-metadata-oai-snapshot.json arxiv.zip; \
		echo "$(GREEN)✅ Локальну інфраструктуру повністю знищено. Пам'ять звільнено!$(RESET)"; \
	fi

# Хак для ігнорування невідомих аргументів
%:
	@:
