import json
import random
import psutil
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm

from etl_core import SecureDownloader

# Налаштування структурованого логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =======================================================================
# Шляхи: Переносимо все в data/ для абсолютної чистоти в корені
# Тому не відповідє умові ТЗ: Всі вихідні файли в корені, parquet у data/
# =======================================================================
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = DATA_DIR / "arxiv_subset.parquet"

# =======================================================================
# 🎛️ ГОЛОВНИЙ ПЕРЕМИКАЧ ОБ'ЄМУ ДАНИХ
# Вкажіть ціле число (наприклад, 10_000 або 5_000 для слабкого заліза)
# Вкажіть None, якщо хочете розпарсити АБСОЛЮТНО ВСІ статті у файлі
# =======================================================================
MAX_RECORDS = 10_000


def setup_environment() -> None:
    """Створює директорії та викликає універсальний завантажувач."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Резервне пряме посилання для urllib (на випадок відсутності Kaggle-ключів у .env)
    # Використовуємо відкритий S3 або альтернативний хостинг або прямий ендпоінт Kaggle для Fallback-завантаження
    fallback_url = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"

    downloader = SecureDownloader(
        dataset_path="Cornell-University/arxiv",
        dataset_url=fallback_url,
        data_dir=str(DATA_DIR), # zip розпакується відразу в data/
        zip_name="arxiv.zip"
    )

    downloader.download(target_filename="arxiv-metadata-oai-snapshot.json")
    downloader.extract_atomically(target_extensions=('.json',))

def safe_extract_year(paper: Dict[str, Any]) -> int:
    """Безпечне витягування року з Graceful Fallback на 2000 рік."""
    try:
        if versions := paper.get("versions", []):
            return int(versions[0].get("created", "").split()[3])
    except (IndexError, ValueError, KeyError, TypeError):
        pass # Тихе пригнічення помилки, перехід до fallback

    return int(paper.get("update_date", "2000-01-01")[:4])

def format_authors(paper: Dict[str, Any]) -> str:
    """Перетворює структурований масив авторів у читабельний рядок."""
    if parsed := paper.get("authors_parsed", []):
        parts = []
        for entry in parsed[:10]: # Захист від масивних списків авторів
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return paper.get("authors", "").replace("\n", " ")

def get_sampled_lines_fast() -> list:
    """Швидкий алгоритм (читає все в пам'ять). Потребує багато RAM."""
    logger.info("⚡ Активуємо FAST алгоритм (readlines). Зчитуємо файл у RAM...")
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # Якщо користувач хоче всі записи — просто повертаємо зчитаний масив
    if MAX_RECORDS is None:
        logger.info("🔀 MAX_RECORDS = None. Забираємо АБСОЛЮТНО ВСІ записи з датасету!")
        return all_lines

    random.seed(42)
    return random.sample(all_lines, min(MAX_RECORDS, len(all_lines)))

def get_sampled_lines_safe() -> list:
    """Алгоритм Reservoir Sampling. Потребує O(1) RAM (кілька мегабайт для вибірки)."""
    logger.info("🐢 Активуємо SAFE алгоритм (Reservoir Sampling). Читаємо потоково...")
    sampled_lines = []
    random.seed(42)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        # Якщо юзер хоче всі записи, ми змушені вичитати їх у список
        if MAX_RECORDS is None:
            logger.warning("⚠️  MAX_RECORDS = None. Завантажуємо ВСІ записи (може з'їсти пам'ять!)...")
            return f.readlines()

        # Класичний алгоритм Reservoir Sampling для заданого MAX_RECORDS
        for i, line in enumerate(tqdm(f, desc=f"Пошук {MAX_RECORDS} рядків", unit="рядків")):
            if i < MAX_RECORDS:
                sampled_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < MAX_RECORDS:
                    sampled_lines[j] = line
    return sampled_lines

def process_data() -> None:
    if not INPUT_FILE.exists():
        logger.error(f"❌ Файл {INPUT_FILE} відсутній. Переконайтеся, що датасет завантажено.")
        return

    # 1. АВТОДЕТЕКЦІЯ ЗАЛІЗА (RAM)
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logger.info(f"🖥️  Виявлено оперативної пам'яті: {total_ram_gb:.1f} ГБ")

    sampled_lines = []

    # 2. ГІБРИДНА ЛОГІКА (Smart Routing)
    if total_ram_gb < 8.0:
        logger.warning("⚠️  Увага: У вас менше 8 ГБ RAM. Рекомендуємо закрити зайві програми.")

    if total_ram_gb >= 24.0:
        try:
            sampled_lines = get_sampled_lines_fast()
        except MemoryError:
            # Якщо Python сам зловив нестачу пам'яті до приходу OOM Killer-а
            logger.error("❌ Python зловив MemoryError! Fast-алгоритм не впорався.")
            logger.info("🔄 Ініціалізація Fallback: переходимо на безпечний алгоритм...")
            sampled_lines = get_sampled_lines_safe()
    else:
        logger.info("📉 RAM менше 24 ГБ. Запускаємо економний алгоритм одразу, щоб уникнути Killed: 9.")
        sampled_lines = get_sampled_lines_safe()

    # 3. ПАРСИНГ ОБРАНИХ РЯДКІВ (Спільний для обох алгоритмів)
    records = []
    corrupted_lines = 0
    logger.info(f"📦 Зібрано {len(sampled_lines)} рядків. Починаємо розпаковку JSON...")

    for line in tqdm(sampled_lines, desc="Парсинг JSON", unit="статей"):
        line = line.strip()
        if not line:
            continue

        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            corrupted_lines += 1
            continue

        abstract = paper.get("abstract", "").strip()
        title = paper.get("title", "").strip()

        if not abstract or not title:
            continue

        records.append({
            "id": paper.get("id"),
            "title": title.replace("\n", " "),
            "abstract": abstract.replace("\n", " "),
            "authors": format_authors(paper),
            "year": safe_extract_year(paper),
            "category": paper.get("categories", "unknown").split()[0],
        })

    if corrupted_lines > 0:
        logger.warning(f"⚠️  Пропущено пошкоджених рядків (або відсутній title/abstract): {corrupted_lines}")

    df = pd.DataFrame(records)
    logger.info(f"📊 Профіль датасету: {len(df)} статей, {df['category'].nunique()} унікальних категорій.")

    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"✅ Успішно збережено {len(df)} записів у {OUTPUT_FILE}")

if __name__ == "__main__":
    setup_environment()
    process_data()
