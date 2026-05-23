import os
import zipfile
import shutil
import urllib.request
import urllib.error
import sys
import logging

# Налаштування логера для повідомлень (прогрес-бар використовує sys.stdout)
logger = logging.getLogger(__name__)

# Спробуємо імпортувати Kaggle API. Якщо бібліотеки немає, ми просто перейдемо на urllib
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_LIB_AVAILABLE = True
except ImportError:
    KAGGLE_LIB_AVAILABLE = False


class SecureDownloader:
    def __init__(self, dataset_path, dataset_url=None, data_dir=".", zip_name="arxiv.zip"):
        """
        Ініціалізує гібридний завантажувач (Kaggle API + urllib Fallback)
        dataset_path: шлях для API (напр. 'Cornell-University/arxiv')
        dataset_url: пряме посилання для urllib (напр. S3 bucket url)
        """
        self.dataset_path = dataset_path
        self.dataset_url = dataset_url
        self.data_dir = data_dir
        self.zip_path = os.path.join(self.data_dir, zip_name)

        if self.data_dir != ".":
            os.makedirs(self.data_dir, exist_ok=True)

    def is_valid_zip(self):
        """Перевірка цілісності ZIP-архіву (захист від HTML-заглушок та битих файлів)"""
        if not os.path.exists(self.zip_path) or not zipfile.is_zipfile(self.zip_path):
            return False
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                if z.testzip() is not None:
                    return False
        except Exception:
            return False
        return True

    def download_progress(self, count, block_size, total_size):
        """Візуалізація прогресу в консолі для urllib завантаження"""
        if total_size > 0:
            percent = min(int(count * block_size * 100 / total_size), 100)
            bar = '█' * int(30 * percent / 100) + '░' * (30 - int(30 * percent / 100))
            mb = (count * block_size) / 1048576
            tot_mb = total_size / 1048576
            sys.stdout.write(f"\r                               | 📥 [{bar}] {percent}% | {mb:.1f}/{tot_mb:.1f} MB")
            sys.stdout.flush()
        else:
            kb = (count * block_size) / 1024
            sys.stdout.write(f"\r                               | 📥 Завантажено: {kb:.1f} KB")
            sys.stdout.flush()

    def _download_via_api(self):
        """Внутрішній метод для завантаження через офіційне API Kaggle."""
        logger.info(f"🤖 Виявлено ключі Kaggle. Ініціалізація офіційного API...")
        logger.warning(f"⏳ УВАГА: Розмір датасету ArXiv становить ~1.3 ГБ!")
        logger.info(f"⏳ Залежно від швидкості інтернету завантаження може тривати 5-15 хвилин. Зачекайте...")
        logger.info(f"⏳ Завантаження датасету '{self.dataset_path}' у '{self.data_dir}'...")

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_path, path=self.data_dir, unzip=False)

        # Перейменовуємо скачаний файл у наш стандартний zip_path
        downloaded_files = [f for f in os.listdir(self.data_dir) if f.endswith('.zip') and f != os.path.basename(self.zip_path)]
        if downloaded_files:
            os.rename(os.path.join(self.data_dir, downloaded_files[0]), self.zip_path)

    def _download_via_urllib(self):
        """Внутрішній метод для завантаження через urllib із підтримкою HTTP Range (дозавантаження)."""
        if not self.dataset_url:
            raise Exception("Fallback URL не вказано, а ключі Kaggle відсутні.")

        existing_size = 0
        if os.path.exists(self.zip_path):
            existing_size = os.path.getsize(self.zip_path)

        req = urllib.request.Request(self.dataset_url, headers={'User-Agent': 'Mozilla/5.0'})

        # Якщо файл вже частково завантажено, просимо сервер віддати лише залишок
        if existing_size > 0:
            req.add_header('Range', f'bytes={existing_size}-')
            logger.info(f"🌐 Ключі Kaggle відсутні. Відновлення завантаження (urllib) з {existing_size / 1024 / 1024:.1f} MB...")
        else:
            logger.info("🌐 Ключі Kaggle відсутні. Ініціалізація прямого завантаження (urllib)...")

        logger.info(f"⏳ Завантаження за посиланням: {self.dataset_url}")

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                content_length = int(response.headers.get('Content-Length', -1))

                # Обробка 206 Partial Content (Сервер підтримує дозавантаження)
                if response.status == 206:
                    expected_total = existing_size + content_length if content_length != -1 else -1
                    mode = 'ab'  # Append binary (дозапис)
                    downloaded_bytes = existing_size
                    logger.info("🔄 Сервер підтримує Range-запити. Продовжуємо завантаження...")
                else:
                    expected_total = content_length
                    mode = 'wb'  # Write binary (з нуля)
                    downloaded_bytes = 0
                    if existing_size > 0:
                        logger.warning("⚠️  Сервер не підтримує відновлення (Range). Починаємо завантаження з нуля...")

                block_size = 8192
                count = 0

                with open(self.zip_path, mode) as out_file:
                    # Якщо качаємо з нуля, count починається з 0
                    # Якщо дозаписуємо, вираховуємо стартовий count для коректної роботи self.download_progress
                    if downloaded_bytes > 0:
                        count = downloaded_bytes // block_size

                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        count += 1

                        # Передаємо поточний стан у ваш кастомний прогрес-бар
                        self.download_progress(count, block_size, expected_total)

                    print()  # Новий рядок після прогрес-бару

        except urllib.error.HTTPError as e:
            # 416 означає, що локальний файл більший або не збігається з сервером
            if e.code == 416:
                logger.warning("⚠️  Помилка 416 (Range Not Satisfiable). Локальний файл конфліктує із сервером. Видаляємо та качаємо наново...")
                if os.path.exists(self.zip_path):
                    os.remove(self.zip_path)
                self._download_via_urllib()  # Рекурсивний ретрай з нуля
            else:
                raise e

    def download(self, target_filename="arxiv-metadata-oai-snapshot.json"):
        """Головний метод завантаження з розумним маршрутизатором (Smart Router)"""
        logger.info("🔍 Перевірка локальних файлів...")

        # 1. Idempotency: Якщо цільовий файл вже є, нічого не качаємо
        target_file = os.path.join(self.data_dir, target_filename)
        if os.path.exists(target_file):
            logger.info(f"🔋 Знайдено готовий файл: {target_filename}. Пропускаємо завантаження.")
            return

        # 2. Idempotency: Якщо архів вже є і він цілий
        if os.path.exists(self.zip_path):
            if self.is_valid_zip():
                logger.info("🔋 Архів цілий. Пропускаємо мережевий запит.")
                return
            else:
                # Залишаємо архів для Range-запиту
                logger.warning("🪫 Архів неповний або пошкоджений. Спроба відновлення завантаження...")

        # 3. Smart Router: Визначаємо спосіб завантаження
        has_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

        try:
            if has_credentials and KAGGLE_LIB_AVAILABLE:
                self._download_via_api()
            else:
                self._download_via_urllib()

            # Фінальна перевірка. Якщо і після скачування файл битий — значить це HTML-заглушка або битий архів
            if not self.is_valid_zip():
                os.remove(self.zip_path)
                raise Exception("Завантажений файл пошкоджено на етапі передачі або це HTML-заглушка Kaggle. Файл видалено.")

            logger.info("✅ Завантаження завершено успішно.")

        except Exception as e:
            logger.error(f"\n⚠️  ПОМИЛКА ЗАВАНТАЖЕННЯ: {e}")
            logger.info("🔄 Активуємо Fallback (Резервний план):")
            logger.info(f"   👉 1. Завантажте датасет вручну: https://www.kaggle.com/datasets/{self.dataset_path}")
            logger.info(f"   👉 2. Покладіть завантажений arxiv.zip у корінь проекту")
            logger.info(f"   👉 3. Перезапустіть 'make etl'")
            sys.exit(1)

    def extract_atomically(self, target_extensions=('.json',), expected_filename="arxiv-metadata-oai-snapshot.json"):
        """Атомарне розпакування з Flattening (вирівнюванням директорій)"""

        # Відновлена логіка ідемпотентності
        target_file = os.path.join(self.data_dir, expected_filename)
        if os.path.exists(target_file):
            return [target_file]

        if not self.is_valid_zip():
            raise Exception("❌ Критична помилка: Архів відсутній або пошкоджений.")

        extracted_files = []
        logger.info(f"📦 Аналізуємо вміст архіву (це може зайняти хвилину для 1.3 ГБ)...")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            data_files = [f for f in zip_ref.namelist() if f.endswith(target_extensions)]

            if not data_files:
                raise Exception(f"В архіві немає файлів з розширеннями {target_extensions}!")

            for file_in_zip in data_files:
                # Flattening: ігноруємо вкладені папки всередині ZIP
                final_path = os.path.join(self.data_dir, os.path.basename(file_in_zip))
                tmp_extract_path = final_path + ".tmp_extract"

                if os.path.exists(final_path):
                    logger.info(f"⚡ Файл '{os.path.basename(final_path)}' вже готовий. Пропускаємо.")
                    extracted_files.append(final_path)
                    continue

                try:
                    logger.info(f"   ⚙️  Витягуємо '{os.path.basename(file_in_zip)}' атомарно...")
                    with zip_ref.open(file_in_zip) as source, open(tmp_extract_path, "wb") as target:
                        shutil.copyfileobj(source, target)

                    os.replace(tmp_extract_path, final_path) # Атомарний коміт на диск
                    extracted_files.append(final_path)

                except Exception as extract_err:
                    raise Exception(f"Помилка фізичного запису на диск: {extract_err}")
                finally:
                    if os.path.exists(tmp_extract_path):
                        os.remove(tmp_extract_path)

        logger.info("✅ Успіх! Файли витягнуто безпечно.")
        return extracted_files
