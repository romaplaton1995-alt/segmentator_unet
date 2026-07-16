from __future__ import annotations

import json
import logging
import os
import shutil
import traceback
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from inference import (
    MODALITIES,
    get_model,
    load_nifti,
    run_inference,
)
from mesh_utils import build_glb


logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger("main")

app = FastAPI(
    title="BraTS Tumor Segmentation API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

TEMP_DIR = Path("/app/temp")
OUTPUT_DIR = Path("/app/output")

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_upload_file(
    upload: UploadFile,
    path: Path,
) -> None:
    with path.open("wb") as output:
        shutil.copyfileobj(
            upload.file,
            output,
        )


def extension_for_upload(
    filename: str | None,
) -> str:
    if not filename:
        return ".nii.gz"

    name = filename.lower()

    if name.endswith(".nii.gz"):
        return ".nii.gz"

    if name.endswith(".nii"):
        return ".nii"

    raise ValueError(
        "Поддерживаются только .nii и .nii.gz."
    )


def metadata_path(
    session_id: str,
) -> Path:
    return OUTPUT_DIR / f"{session_id}.json"


def volume_path(
    session_id: str,
) -> Path:
    return OUTPUT_DIR / f"{session_id}_flair.npy"


def glb_path(
    session_id: str,
) -> Path:
    return OUTPUT_DIR / f"{session_id}.glb"


def calculate_class_voxels(
    mask: np.ndarray,
) -> dict[str, int]:
    return {
        "background": int(
            np.sum(mask == 0)
        ),
        "necrosis": int(
            np.sum(mask == 1)
        ),
        "edema": int(
            np.sum(mask == 2)
        ),
        "enhancing": int(
            np.sum(mask == 3)
        ),
        "whole_tumor": int(
            np.sum(mask > 0)
        ),
        "tumor_core": int(
            np.sum(np.isin(mask, [1, 3]))
        ),
    }


@app.on_event("startup")
def startup() -> None:
    logger.info(
        "Загрузка модели при запуске backend."
    )
    get_model()
    logger.info(
        "Backend готов."
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "BraTS Tumor Segmentation API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(
            get_model().outc.weight.device
        ),
    }


@app.get(
    "/glb/{session_id}",
)
def get_glb(
    session_id: str,
) -> FileResponse:
    path = glb_path(session_id)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="GLB-файл не найден.",
        )

    return FileResponse(
        path=str(path),
        media_type="model/gltf-binary",
        filename="tumor_reconstruction.glb",
    )


@app.get(
    "/volume/{session_id}",
)
def get_volume(
    session_id: str,
) -> dict:
    path = volume_path(session_id)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="FLAIR-объём не найден.",
        )

    data = np.load(path).astype(
        np.float32
    )

    minimum = float(data.min())
    maximum = float(data.max())

    if maximum > minimum:
        normalized = (
            (data - minimum)
            / (maximum - minimum)
            * 255.0
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(
            data,
            dtype=np.uint8,
        )

    return {
        "shape": [
            int(value)
            for value in normalized.shape
        ],
        "data": normalized.flatten().tolist(),
    }


@app.get(
    "/metadata/{session_id}",
)
def get_metadata(
    session_id: str,
) -> dict:
    path = metadata_path(session_id)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Метаданные не найдены.",
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@app.delete(
    "/session/{session_id}",
)
def delete_session(
    session_id: str,
) -> dict:
    paths = [
        glb_path(session_id),
        volume_path(session_id),
        metadata_path(session_id),
    ]

    for path in paths:
        try:
            path.unlink(
                missing_ok=True,
            )
        except OSError as exc:
            logger.warning(
                "Не удалось удалить %s: %s",
                path,
                exc,
            )

    return {
        "status": "deleted",
        "session_id": session_id,
    }


@app.post(
    "/predict",
)
async def predict(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
) -> JSONResponse:
    session_id = str(
        uuid.uuid4()
    )

    session_dir = (
        TEMP_DIR / session_id
    )
    session_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    uploads = {
        "flair": flair,
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
    }

    file_paths: dict[str, str] = {}

    try:
        for modality in MODALITIES:
            upload = uploads[modality]
            extension = extension_for_upload(
                upload.filename
            )

            path = session_dir / (
                f"{modality}{extension}"
            )

            save_upload_file(
                upload,
                path,
            )

            file_paths[modality] = str(path)

        mask, affine, flair_raw, spacing = (
            run_inference(
                file_paths=file_paths,
                session_dir=str(session_dir),
            )
        )

        output_glb = glb_path(
            session_id
        )

        build_glb(
            pred_mask=mask,
            flair_raw=flair_raw,
            output_path=output_glb,
        )

        np.save(
            volume_path(session_id),
            flair_raw.astype(
                np.float32
            ),
        )

        class_voxels = calculate_class_voxels(
            mask
        )

        metadata = {
            "session_id": session_id,
            "shape": [
                int(value)
                for value in mask.shape
            ],
            "spacing": [
                float(value)
                for value in spacing
            ],
            "modalities": list(
                MODALITIES
            ),
            "classes": {
                "0": "background",
                "1": "necrosis",
                "2": "edema",
                "3": "enhancing",
            },
            "class_voxels": class_voxels,
            "glb_url": (
                f"/glb/{session_id}"
            ),
            "volume_url": (
                f"/volume/{session_id}"
            ),
            "metadata_url": (
                f"/metadata/{session_id}"
            ),
        }

        with metadata_path(
            session_id
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Анализ завершён: %s",
            session_id,
        )

        return JSONResponse(
            content=metadata,
            headers={
                "X-Session-Id": session_id,
            },
        )

    except Exception as exc:
        logger.error(
            "Ошибка сессии %s: %s",
            session_id,
            exc,
        )
        logger.error(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        shutil.rmtree(
            session_dir,
            ignore_errors=True,
        )
