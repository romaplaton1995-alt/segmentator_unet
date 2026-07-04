import numpy as np
from skimage import measure
from scipy import ndimage
import trimesh


def extract_brain_mask(flair_volume, threshold_percentile=15):
    nonzero = flair_volume[flair_volume != 0]
    if nonzero.size == 0:
        return np.zeros_like(flair_volume, dtype=np.uint8)
    mask = flair_volume > np.percentile(nonzero, threshold_percentile)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    labeled, num = ndimage.label(mask)
    if num > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        mask = labeled == (np.argmax(sizes) + 1)
    return mask.astype(np.uint8)


def clean_tumor_mask(mask, min_size=50):
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep = np.where(sizes >= min_size)[0] + 1
    return np.isin(labeled, keep).astype(np.uint8)


def mask_to_mesh(mask, level=0.5, step_size=1):
    if mask.sum() == 0:
        return None
    verts, faces, normals, _ = measure.marching_cubes(mask, level=level, step_size=step_size)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def safe_decimate(mesh, target_faces):
    """Пытается уменьшить количество полигонов. Если backend недоступен — возвращает меш как есть."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except ImportError:
        print(f"⚠️ Децимация пропущена (нет backend'а), меш останется с {len(mesh.faces)} гранями")
        return mesh


def build_glb(pred_mask, flair_raw, output_path):
    brain_mask = extract_brain_mask(flair_raw)
    tumor_mask = clean_tumor_mask(pred_mask, min_size=30)

    brain_mesh = mask_to_mesh(brain_mask, step_size=2)
    tumor_mesh = mask_to_mesh(tumor_mask, step_size=1)

    meshes = []
    if brain_mesh is not None:
        brain_mesh = trimesh.smoothing.filter_laplacian(brain_mesh, iterations=5)
        brain_mesh = safe_decimate(brain_mesh, 50000)
        brain_mesh.visual.face_colors = [200, 200, 200, 90]
        meshes.append(brain_mesh)

    if tumor_mesh is not None:
        tumor_mesh = trimesh.smoothing.filter_laplacian(tumor_mesh, iterations=3)
        tumor_mesh = safe_decimate(tumor_mesh, 20000)
        tumor_mesh.visual.face_colors = [220, 30, 30, 255]
        meshes.append(tumor_mesh)

    if not meshes:
        raise ValueError("Не удалось построить ни одного меша — маски пустые")

    # В функции build_glb перед scene = trimesh.Scene(meshes)
    for m in meshes:
        m.fix_normals()  # Исправляет направление граней, чтобы они были видны

    scene = trimesh.Scene(meshes)
    scene.export(output_path)
    return output_path

