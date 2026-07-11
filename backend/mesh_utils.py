import numpy as np
from skimage import measure
from scipy import ndimage
import trimesh
import logging

logger = logging.getLogger("main")


def extract_brain_mask(flair_volume, threshold_percentile=15):
    """Извлекает маску мозга из FLAIR объема."""
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


def clean_tumor_mask(mask, min_size=10):  # УМЕНЬШИЛИ с 30/20/30 до 10
    """Очищает маску опухоли, удаляя очень мелкие компоненты."""
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep = np.where(sizes >= min_size)[0] + 1
    return np.isin(labeled, keep).astype(np.uint8)


def mask_to_mesh(mask, level=0.5, step_size=1):
    """Конвертирует бинарную маску в 3D меш."""
    if mask.sum() == 0:
        return None

    try:
        verts, faces, normals, _ = measure.marching_cubes(mask, level=level, step_size=step_size)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # Исправляем нормали - они должны быть наружу
        mesh.fix_normals()
        return mesh
    except Exception as e:
        logger.error(f"Ошибка создания меша: {e}")
        return None


def safe_decimate(mesh, target_faces):
    """Пытается уменьшить количество полигонов."""
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except ImportError:
        logger.info(f"⚠️ Децимация пропущена (нет backend'а), меш останется с {len(mesh.faces)} гранями")
        return mesh
    except Exception as e:
        logger.error(f"Ошибка децимации: {e}")
        return mesh


def build_glb(pred_mask, flair_raw, output_path):
    """
    Строит GLB файл с 3D моделью мозга и опухоли.
    """
    brain_mask = extract_brain_mask(flair_raw)

    # Маски для каждого класса
    necrosis_mask = (pred_mask == 1).astype(np.uint8)
    edema_mask = (pred_mask == 2).astype(np.uint8)
    enhancing_mask = (pred_mask == 3).astype(np.uint8)

    logger.info(f"Класс 1 (некроз): {necrosis_mask.sum()} вокселей")
    logger.info(f"Класс 2 (отёк): {edema_mask.sum()} вокселей")
    logger.info(f"Класс 3 (активная опухоль): {enhancing_mask.sum()} вокселей")

    # Очистка с меньшим min_size
    necrosis_mask = clean_tumor_mask(necrosis_mask, min_size=10)
    edema_mask = clean_tumor_mask(edema_mask, min_size=10)
    enhancing_mask = clean_tumor_mask(enhancing_mask, min_size=10)

    logger.info(
        f"После очистки - некроз: {necrosis_mask.sum()}, отёк: {edema_mask.sum()}, активная опухоль: {enhancing_mask.sum()}")

    # Создаём меши
    brain_mesh = mask_to_mesh(brain_mask, step_size=2)
    necrosis_mesh = mask_to_mesh(necrosis_mask, step_size=1)
    edema_mesh = mask_to_mesh(edema_mask, step_size=1)
    enhancing_mesh = mask_to_mesh(enhancing_mask, step_size=1)

    # Создаём сцену с явными именами
    scene = trimesh.Scene()

    # 1. Мозг (полупрозрачный серый) - рендерится ПЕРВЫМ
    if brain_mesh is not None:
        brain_mesh = trimesh.smoothing.filter_laplacian(brain_mesh, iterations=5)
        brain_mesh = safe_decimate(brain_mesh, 50000)
        brain_mesh.visual.face_colors = [200, 200, 200, 60]  # Полупрозрачный
        scene.add_geometry(brain_mesh, node_name='brain', geom_name='brain')
        logger.info("Добавлен меш: мозг")

    # 2. Отёк (голубой, очень прозрачный) - рендерится ВТОРЫМ
    if edema_mesh is not None:
        edema_mesh = trimesh.smoothing.filter_laplacian(edema_mesh, iterations=3)
        edema_mesh = safe_decimate(edema_mesh, 30000)
        edema_mesh.visual.face_colors = [100, 180, 255, 80]  # Более прозрачный
        scene.add_geometry(edema_mesh, node_name='edema', geom_name='edema')
        logger.info("Добавлен меш: отёк")

    # 3. Некроз (оранжевый, средняя прозрачность) - рендерится ТРЕТЬИМ
    if necrosis_mesh is not None:
        necrosis_mesh = trimesh.smoothing.filter_laplacian(necrosis_mesh, iterations=3)
        necrosis_mesh = safe_decimate(necrosis_mesh, 15000)
        necrosis_mesh.visual.face_colors = [255, 165, 0, 180]  # Менее прозрачный
        scene.add_geometry(necrosis_mesh, node_name='necrosis', geom_name='necrosis')
        logger.info("Добавлен меш: некроз")

    # 4. Активная опухоль (ярко-красный, непрозрачный) - рендерится ПОСЛЕДНИМ
    if enhancing_mesh is not None:
        enhancing_mesh = trimesh.smoothing.filter_laplacian(enhancing_mesh, iterations=3)
        enhancing_mesh = safe_decimate(enhancing_mesh, 10000)
        enhancing_mesh.visual.face_colors = [255, 30, 30, 255]  # Непрозрачный
        scene.add_geometry(enhancing_mesh, node_name='enhancing', geom_name='enhancing')
        logger.info("Добавлен меш: активная опухоль")

    if len(scene.geometry) == 0:
        raise ValueError("Не удалось построить ни одного меша — маски пустые")

    scene.export(output_path)
    logger.info(f"GLB экспортирован: {output_path}")

    return output_path