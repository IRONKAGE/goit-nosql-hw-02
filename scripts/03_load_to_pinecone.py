import os
import time
import logging
from pathlib import Path
from functools import wraps
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Налаштування індустріального структурованого логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
load_dotenv()

# Визначення строгої типізації шляхів через Pathlib
INPUT_PARQUET = Path("data/arxiv_subset.parquet")
INPUT_EMBEDDINGS = Path("embeddings/embeddings.npy")
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 100 # Оптимальний ліміт пакета для стабільної обробки API Pinecone

# =====================================================================
# 🛡️ FAULT TOLERANCE: Експоненційний бекофф
# =====================================================================
def with_exponential_backoff(max_retries: int = 5, base_delay: float = 1.0):
    """
    Декоратор для захисту мережевих запитів. Автоматично обробляє помилки
    лімітів (429 Too Many Requests) та таймаути, збільшуючи паузу: 1с, 2с, 4с, 8с...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"❌ Досягнуто ліміту спроб ({max_retries}) для {func.__name__}.")
                        raise e
                    sleep_time = base_delay * (2 ** (retries - 1))
                    logger.warning(f"⚠️  Збій API: {str(e)}. Спроба {retries}/{max_retries} за {sleep_time}с...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

# =====================================================================

@with_exponential_backoff(max_retries=3)
def create_index_if_missing(pc_client: Pinecone) -> None:
    """Створює індекс, сумісний з pinecone-client 4.1.0."""

    # 1. Офіційний та найчистіший спосіб для версії 4.1.0 - list_indexes().names()
    existing_indexes = pc_client.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        logger.info(f"🛠️  Створення індексу '{INDEX_NAME}' (Метрика: Cosine, Простір: {VECTOR_DIM})...")

        # 2. Офіційна документація: spec є обов'язковим об'єктом SDK v4+
        pc_client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        # 3. Універсальна перевірка статусу ready (підтримує як об'єкти, так і словники)
        ready = False
        while not ready:
            status = pc_client.describe_index(INDEX_NAME).status
            ready = getattr(status, 'ready', False) if not isinstance(status, dict) else status.get('ready', False)

            if not ready:
                logger.info("⏳ Очікування конфігурації та підняття реплік індексу...")
                time.sleep(2)

        logger.info(f"✅ Індекс '{INDEX_NAME}' успішно створено та готовий до роботи!")
    else:
        logger.info(f"🔋 Індекс '{INDEX_NAME}' вже присутній у системі. Пропускаємо створення.")

@with_exponential_backoff(max_retries=5)
def safe_upsert(index_client, vectors: List[Dict[str, Any]]) -> None:
    """Атомарне пакетне завантаження масиву векторів з автоматичними ретраями."""
    index_client.upsert(vectors=vectors)

def main():
    if not INPUT_PARQUET.exists() or not INPUT_EMBEDDINGS.exists():
        logger.error("❌ Критичні інфраструктурні артефакти відсутні! Спочатку виконайте кроки 01 та 02.")
        return

    # Розумна маршрутизація (Smart Routing)
    env_mode = os.environ.get("ACTIVE_ENV", "local").strip().lower()
    if env_mode == "local":
        local_host = os.environ.get("PINECONE_LOCAL_HOST", "http://127.0.0.1:5080")
        logger.info(f"🖥️  Активація режиму LOCAL EMULATOR. Хост: {local_host}")
        pc = Pinecone(api_key="local-dummy-key", host=local_host)
    else:
        logger.info("☁️  Активація режиму REAL CLOUD SAAS (Pinecone Production)...")
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Створення та перевірка індексу
    create_index_if_missing(pc)

    # Жорстко перенаправляємо Data Plane на наш локальний порт, ігноруючи внутрішні роути емулятора
    if env_mode == "local":
        raw_host = pc.describe_index(INDEX_NAME).host
        local_url = f"http://{raw_host}" if not raw_host.startswith("http") else raw_host.replace("https", "http")
        local_url = local_url.replace("0.0.0.0", "127.0.0.1").replace("localhost", "127.0.0.1")
        index = pc.Index(name=INDEX_NAME, host=local_url)
    else:
        index = pc.Index(name=INDEX_NAME)

    logger.info("📖 Завантаження обробленого Parquet-датасету та матриці NumPy в RAM...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    if len(df) != len(embeddings):
        logger.error(f"❌ Критична розсинхронізація: рядків у Parquet ({len(df)}) не збігається з кількістю ембеддингів ({len(embeddings)})!")
        return

    logger.info(f"🚀 Початок потокового Upsert-завантаження батчами по {BATCH_SIZE} векторів...")

    # Обробка даних пакетами для стабільності API
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_emb = embeddings[i:i+BATCH_SIZE]

        vectors_to_upsert = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            # Жорстка типізація, кастинг та обрізка метаданих для захисту ліміту в 40 КБ
            vectors_to_upsert.append({
                "id": f"paper_{row['id']}",
                "values": batch_emb[j].tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title": str(row["title"])[:250],
                    "abstract": str(row["abstract"])[:500],
                    "authors": str(row["authors"])[:200],
                    "year": int(row["year"]),
                    "category": str(row["category"])
                }
            })

        # Безпечний виклик мережевої функції
        safe_upsert(index, vectors_to_upsert)

    # 🔥 ПОВНЕ ВІДНОВЛЕННЯ ТА ПРОКАЧКА ФІНАЛЬНОЇ ВАЛІДАЦІЇ СТАТИСТИКИ БАЗИ ДАНИХ
    logger.info("🔄 Очікування фінальної індексації та синхронізації метаданих рушієм...")
    time.sleep(2)

    stats = index.describe_index_stats()
    logger.info(f"=========================================================================")
    logger.info(f"✅ ЗАВАНТАЖЕННЯ ЗАВЕРШЕНО УСПІШНО!")
    logger.info(f"📊 Фінальний статус індексу '{INDEX_NAME}':")
    logger.info(f"   - Загальна кількість векторів у базі: {stats.total_vector_count}")
    logger.info(f"   - Розмірність простору: {stats.dimension}")
    logger.info(f"=========================================================================")

if __name__ == "__main__":
    main()
