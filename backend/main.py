from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import shutil
import traceback
import logging

from inference import run_inference, get_model
from mesh_utils import build_glb


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="BraTS Tumor Segmentation API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ПАПКИ ДЛЯ ВРЕМЕННЫХ И ВЫХОДНЫХ ФАЙЛОВ
# ============================================================

TEMP_DIR = "/app/temp"
OUTPUT_DIR = "/app/output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_upload_extension(filename: str) -> str:
    """
    Определяет расширение загруженного файла.

    Поддерживаем:
    - .zip для DICOM-серий
    - .nii для NIfTI
    - .nii.gz для сжатого NIfTI

    Если расширение неизвестно, по умолчанию считаем файл NIfTI.
    """
    if filename is None:
        return ".nii"

    filename = filename.lower()

    if filename.endswith(".zip"):
        return ".zip"

    if filename.endswith(".nii.gz"):
        return ".nii.gz"

    if filename.endswith(".nii"):
        return ".nii"

    return ".nii"


def save_upload_file(upload: UploadFile, path: str):
    """
    Сохраняет UploadFile на диск.
    """
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def load_model_on_startup():
    """
    Загружаем модель при старте контейнера, чтобы первый запрос
    не тратил время на инициализацию весов.
    """
    logger.info("Загрузка модели при старте backend...")
    get_model()
    logger.info("Модель успешно загружена.")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    """
    Корневой endpoint.
    Нужен, чтобы при переходе на http://localhost:8000/
    backend не возвращал 404.
    """
    return {
        "service": "BraTS Tumor Segmentation API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Проверка работоспособности backend.
    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    """
    Основной endpoint сегментации.

    Принимает 4 файла:
    - FLAIR
    - T1
    - T1ce
    - T2

    Сейчас поддерживаются:
    - .nii
    - .nii.gz
    - .zip с DICOM-серией внутри

    Возвращает:
    - .glb файл с 3D-моделью мозга и опухоли
    """

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    logger.info(f"Новая сессия обработки: {session_id}")

    file_paths = {}

    try:
        uploads = [
            ("flair", flair),
            ("t1", t1),
            ("t1ce", t1ce),
            ("t2", t2),
        ]

        # ----------------------------------------------------
        # 1. Сохраняем загруженные файлы во временную папку
        # ----------------------------------------------------

        for name, upload in uploads:
            ext = get_upload_extension(upload.filename)
            path = os.path.join(session_dir, f"{name}{ext}")

            logger.info(f"Сохранение файла {name}: {upload.filename} -> {path}")

            save_upload_file(upload, path)
            file_paths[name] = path

        logger.info(f"Все файлы сохранены для сессии {session_id}")

        # ----------------------------------------------------
        # 2. Запускаем инференс модели
        # ----------------------------------------------------

        logger.info(f"Запуск инференса для сессии {session_id}")

        mask, affine, flair_raw = run_inference(
            file_paths=file_paths,
            session_dir=session_dir
        )

        logger.info(
            f"Инференс завершён для сессии {session_id}. "
            f"Размер маски: {mask.shape}, voxels={int(mask.sum())}"
        )

        # ----------------------------------------------------
        # 3. Строим 3D-модель GLB
        # ----------------------------------------------------

        output_path = os.path.join(OUTPUT_DIR, f"{session_id}.glb")

        logger.info(f"Генерация GLB: {output_path}")

        build_glb(mask, flair_raw, output_path)

        if not os.path.exists(output_path):
            raise RuntimeError("GLB-файл не был создан.")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        logger.info(
            f"GLB успешно создан для сессии {session_id}. "
            f"Размер файла: {file_size_mb:.2f} MB"
        )

        # ----------------------------------------------------
        # 4. Возвращаем GLB во frontend
        # ----------------------------------------------------

        return FileResponse(
            output_path,
            media_type="model/gltf-binary",
            filename="tumor_reconstruction.glb"
        )

    except Exception as e:
        logger.error(f"Ошибка в сессии {session_id}: {str(e)}")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # Удаляем только временную папку с входными файлами.
        # output_path НЕ удаляем, потому что FileResponse должен успеть отдать GLB.
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.info(f"Временная папка сессии удалена: {session_dir}")

