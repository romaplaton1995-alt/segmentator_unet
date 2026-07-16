from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from model_unet import UNet3D


logger = logging.getLogger("main")

MODEL_PATH = Path("/app/weights/best_model.pt")
PATCH_SIZE = (96, 96, 96)
OVERLAP = 0.5
NUM_CLASSES = 4

MODALITIES = ("t1", "t1ce", "t2", "flair")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ИСПРАВЛЕНО: инициализируем None, а не пустой моделью
_model: UNet3D | None = None


def _extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if (
        isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    ):
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint не содержит state_dict."
        )

    return state_dict


def get_model() -> UNet3D:
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл весов не найден: {MODEL_PATH}"
        )

    logger.info(
        "Загрузка UNet3D: "
        f"device={DEVICE}, "
        f"weights={MODEL_PATH}"
    )

    model = UNet3D(
        in_channels=4,
        out_channels=NUM_CLASSES,
        base_features=8,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    state_dict = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Весы несовместимы с inference-моделью. "
            "Проверьте base_features=8, "
            "in_channels=4 и out_channels=4."
        ) from exc

    model.to(DEVICE)
    model.eval()

    _model = model

    logger.info("Модель успешно загружена.")

    return _model


def load_nifti(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    path = Path(file_path)

    if not path.name.lower().endswith(
        (".nii", ".nii.gz")
    ):
        raise ValueError(
            f"Ожидался NIfTI-файл: {file_path}"
        )

    image = nib.load(str(path))

    data = image.get_fdata(
        dtype=np.float32
    )

    if data.ndim != 3:
        raise ValueError(
            f"NIfTI должен быть 3D: {file_path}, "
            f"shape={data.shape}"
        )

    affine = image.affine.copy()
    spacing = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]
    )

    # Приводим к соглашению [D, H, W].
    data_dhw = np.transpose(
        data,
        (2, 0, 1),
    )

    return data_dhw, affine, spacing


def validate_modalities(
    volumes: dict[str, np.ndarray],
    spacings: dict[str, tuple[float, ...]],
) -> None:
    shapes = {
        name: tuple(volume.shape)
        for name, volume in volumes.items()
    }

    if len(set(shapes.values())) != 1:
        raise ValueError(
            "Размеры модальностей не совпадают: "
            f"{shapes}"
        )

    reference_spacing = spacings[MODALITIES[0]]

    for name, spacing in spacings.items():
        if not np.allclose(
            spacing,
            reference_spacing,
            atol=1e-3,
        ):
            raise ValueError(
                "Spacing модальностей не совпадает: "
                f"{spacings}"
            )


def zscore_nonzero(
    volume: np.ndarray,
) -> np.ndarray:
    result = volume.astype(
        np.float32,
        copy=True,
    )

    mask = result != 0

    if not np.any(mask):
        return result

    values = result[mask]
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-8:
        result[mask] = 0.0
    else:
        result[mask] = (
            result[mask] - mean
        ) / std

    return result


def load_and_normalize(
    file_paths: dict[str, str],
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    tuple[float, ...],
]:
    volumes: dict[str, np.ndarray] = {}
    spacings: dict[str, tuple[float, ...]] = {}
    affine = np.eye(4, dtype=np.float64)

    for index, modality in enumerate(MODALITIES):
        if modality not in file_paths:
            raise ValueError(
                f"Отсутствует модальность: {modality}"
            )

        volume, current_affine, spacing = load_nifti(
            file_paths[modality]
        )

        if index == 0:
            affine = current_affine

        volumes[modality] = volume
        spacings[modality] = spacing

    validate_modalities(
        volumes=volumes,
        spacings=spacings,
    )

    normalized = []

    for modality in MODALITIES:
        normalized.append(
            zscore_nonzero(volumes[modality])
        )

    image = np.stack(
        normalized,
        axis=0,
    ).astype(np.float32)

    flair_raw = volumes["flair"].astype(
        np.float32,
        copy=True,
    )

    image_tensor = torch.from_numpy(
        image
    ).float()

    return (
        image_tensor,
        affine,
        flair_raw,
        spacings["flair"],
    )


def _steps(
    size: int,
    patch: int,
    stride: int,
) -> list[int]:
    if size <= patch:
        return [0]

    values = list(
        range(
            0,
            size - patch + 1,
            stride,
        )
    )

    last = size - patch

    if values[-1] != last:
        values.append(last)

    return sorted(set(values))


def _pad_image(
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
]:
    _, depth, height, width = image.shape
    pd, ph, pw = patch_size

    pad_d = max(pd - depth, 0)
    pad_h = max(ph - height, 0)
    pad_w = max(pw - width, 0)

    padded = F.pad(
        image,
        (
            0,
            pad_w,
            0,
            pad_h,
            0,
            pad_d,
        ),
    )

    return padded, (pad_d, pad_h, pad_w)


@torch.inference_mode()
def sliding_window_inference(
    model: UNet3D,
    image: torch.Tensor,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    overlap: float = OVERLAP,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(
            "Ожидалась форма image [C,D,H,W]."
        )

    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            "overlap должен быть в диапазоне [0, 1)."
        )

    padded, padding = _pad_image(
        image,
        patch_size,
    )

    _, depth, height, width = padded.shape
    pd, ph, pw = patch_size

    strides = tuple(
        max(
            1,
            int(
                patch * (1.0 - overlap)
            ),
        )
        for patch in patch_size
    )

    z_steps = _steps(
        depth,
        pd,
        strides[0],
    )
    y_steps = _steps(
        height,
        ph,
        strides[1],
    )
    x_steps = _steps(
        width,
        pw,
        strides[2],
    )

    probability_map = torch.zeros(
        (
            NUM_CLASSES,
            depth,
            height,
            width,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    count_map = torch.zeros(
        (
            1,
            depth,
            height,
            width,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    for z in z_steps:
        for y in y_steps:
            for x in x_steps:
                patch = padded[
                    :,
                    z:z + pd,
                    y:y + ph,
                    x:x + pw,
                ].unsqueeze(0).to(DEVICE)

                # ИСПРАВЛЕНО: убрали autocast для избежания конфликта float16/float32
                # Явно приводим к float32 для стабильности
                patch = patch.float()
                logits = model(patch)
                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )[0].float()

                probability_map[
                    :,
                    z:z + pd,
                    y:y + ph,
                    x:x + pw,
                ] += probabilities

                count_map[
                    :,
                    z:z + pd,
                    y:y + ph,
                    x:x + pw,
                ] += 1.0

    probability_map = probability_map / count_map.clamp_min(
        1.0
    )

    pad_d, pad_h, pad_w = padding

    end_d = depth - pad_d if pad_d else depth
    end_h = height - pad_h if pad_h else height
    end_w = width - pad_w if pad_w else width

    return probability_map[
        :,
        :end_d,
        :end_h,
        :end_w,
    ]


def run_inference(
    file_paths: dict[str, str],
    session_dir: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, ...],
]:
    model = get_model()

    image, affine, flair_raw, spacing = (
        load_and_normalize(file_paths)
    )

    probabilities = sliding_window_inference(
        model=model,
        image=image,
    )

    mask = probabilities.argmax(
        dim=0
    ).cpu().numpy().astype(np.uint8)

    logger.info(
        "Инференс завершён: "
        f"shape={mask.shape}, "
        f"classes={np.unique(mask).tolist()}"
    )

    return (
        mask,
        affine,
        flair_raw,
        spacing,
    )
