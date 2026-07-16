from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


logger = logging.getLogger("main")


COLORS = {
    "brain": [200, 200, 200, 55],
    "whole_tumor": [255, 80, 80, 110],
    "tumor_core": [255, 120, 0, 150],
    "necrosis": [255, 165, 0, 190],
    "edema": [100, 180, 255, 100],
    "enhancing": [255, 30, 30, 255],
}


def extract_brain_mask(
    flair_volume: np.ndarray,
    threshold_percentile: float = 15.0,
) -> np.ndarray:
    if flair_volume.ndim != 3:
        raise ValueError(
            "FLAIR должен иметь форму [D,H,W]."
        )

    nonzero = flair_volume[
        np.isfinite(flair_volume)
        & (flair_volume != 0)
    ]

    if nonzero.size == 0:
        return np.zeros_like(
            flair_volume,
            dtype=np.uint8,
        )

    threshold = np.percentile(
        nonzero,
        threshold_percentile,
    )

    mask = (
        np.isfinite(flair_volume)
        & (flair_volume > threshold)
    )

    structure = np.ones(
        (3, 3, 3),
        dtype=bool,
    )

    mask = ndimage.binary_closing(
        mask,
        structure=structure,
    )

    mask = ndimage.binary_fill_holes(mask)

    labeled, count = ndimage.label(mask)

    if count > 1:
        sizes = ndimage.sum(
            mask,
            labeled,
            range(1, count + 1),
        )

        largest_label = int(
            np.argmax(sizes) + 1
        )

        mask = labeled == largest_label

    return mask.astype(np.uint8)


def clean_mask(
    mask: np.ndarray,
    min_size: int = 10,
) -> np.ndarray:
    binary = mask.astype(bool)

    if not binary.any():
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    labeled, count = ndimage.label(
        binary
    )

    if count == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    sizes = ndimage.sum(
        binary,
        labeled,
        range(1, count + 1),
    )

    keep_labels = (
        np.where(
            sizes >= min_size
        )[0]
        + 1
    )

    return np.isin(
        labeled,
        keep_labels,
    ).astype(np.uint8)


def mask_to_mesh(
    mask: np.ndarray,
    *,
    step_size: int = 1,
) -> trimesh.Trimesh | None:
    if mask.ndim != 3:
        raise ValueError(
            "Маска должна иметь форму [D,H,W]."
        )

    if int(mask.sum()) == 0:
        return None

    padded = np.pad(
        mask.astype(np.float32),
        1,
        mode="constant",
    )

    try:
        vertices, faces, normals, _ = (
            measure.marching_cubes(
                padded,
                level=0.5,
                step_size=step_size,
            )
        )

        vertices -= 1.0

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=True,
        )

        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        return mesh

    except Exception as exc:
        logger.exception(
            "Ошибка построения поверхности: %s",
            exc,
        )
        return None


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 1,
) -> trimesh.Trimesh:
    if iterations <= 0:
        return mesh

    try:
        return trimesh.smoothing.filter_laplacian(
            mesh,
            iterations=iterations,
        )
    except Exception as exc:
        logger.warning(
            "Сглаживание пропущено: %s",
            exc,
        )
        return mesh


def safe_decimate(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        simplified = mesh.simplify_quadric_decimation(
            face_count=target_faces
        )

        if simplified is not None:
            return simplified

    except Exception as exc:
        logger.warning(
            "Децимация пропущена: %s",
            exc,
        )

    return mesh


def set_mesh_color(
    mesh: trimesh.Trimesh,
    color: list[int],
) -> None:
    mesh.visual.face_colors = np.tile(
        np.asarray(
            color,
            dtype=np.uint8,
        ),
        (len(mesh.faces), 1),
    )


def add_mesh(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh | None,
    *,
    name: str,
    color: list[int],
    target_faces: int,
    smooth_iterations: int = 1,
) -> None:
    if mesh is None:
        return

    mesh = smooth_mesh(
        mesh,
        iterations=smooth_iterations,
    )

    mesh = safe_decimate(
        mesh,
        target_faces=target_faces,
    )

    set_mesh_color(
        mesh,
        color,
    )

    scene.add_geometry(
        mesh,
        node_name=name,
        geom_name=name,
    )

    logger.info(
        "Добавлен меш %s: vertices=%d, faces=%d",
        name,
        len(mesh.vertices),
        len(mesh.faces),
    )


def build_glb(
    pred_mask: np.ndarray,
    flair_raw: np.ndarray,
    output_path: str | Path,
) -> str:
    if pred_mask.shape != flair_raw.shape:
        raise ValueError(
            "Размеры prediction и FLAIR не совпадают: "
            f"{pred_mask.shape} != {flair_raw.shape}"
        )

    pred_mask = pred_mask.astype(
        np.uint8,
        copy=False,
    )

    brain_mask = extract_brain_mask(
        flair_raw
    )

    masks = {
        "brain": brain_mask,
        "whole_tumor": clean_mask(
            pred_mask > 0,
            min_size=20,
        ),
        "tumor_core": clean_mask(
            np.isin(pred_mask, [1, 3]),
            min_size=10,
        ),
        "necrosis": clean_mask(
            pred_mask == 1,
            min_size=10,
        ),
        "edema": clean_mask(
            pred_mask == 2,
            min_size=10,
        ),
        "enhancing": clean_mask(
            pred_mask == 3,
            min_size=10,
        ),
    }

    for name, mask in masks.items():
        logger.info(
            "%s: %d voxels",
            name,
            int(mask.sum()),
        )

    scene = trimesh.Scene()

    add_mesh(
        scene,
        mask_to_mesh(
            masks["brain"],
            step_size=2,
        ),
        name="brain",
        color=COLORS["brain"],
        target_faces=50000,
        smooth_iterations=1,
    )

    add_mesh(
        scene,
        mask_to_mesh(
            masks["whole_tumor"],
            step_size=1,
        ),
        name="whole_tumor",
        color=COLORS["whole_tumor"],
        target_faces=30000,
        smooth_iterations=1,
    )

    add_mesh(
        scene,
        mask_to_mesh(
            masks["tumor_core"],
            step_size=1,
        ),
        name="tumor_core",
        color=COLORS["tumor_core"],
        target_faces=20000,
        smooth_iterations=1,
    )

    add_mesh(
        scene,
        mask_to_mesh(
            masks["edema"],
            step_size=1,
        ),
        name="edema",
        color=COLORS["edema"],
        target_faces=20000,
        smooth_iterations=1,
    )

    add_mesh(
        scene,
        mask_to_mesh(
            masks["necrosis"],
            step_size=1,
        ),
        name="necrosis",
        color=COLORS["necrosis"],
        target_faces=15000,
        smooth_iterations=1,
    )

    add_mesh(
        scene,
        mask_to_mesh(
            masks["enhancing"],
            step_size=1,
        ),
        name="enhancing",
        color=COLORS["enhancing"],
        target_faces=10000,
        smooth_iterations=1,
    )

    if len(scene.geometry) == 0:
        raise ValueError(
            "Не удалось построить ни одного меша."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene.export(str(output_path))

    logger.info(
        "GLB экспортирован: %s",
        output_path,
    )

    return str(output_path)
