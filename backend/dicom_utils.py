import SimpleITK as sitk
import numpy as np
import os
import zipfile
import shutil


def load_dicom_from_zip(zip_path, extract_dir):
    """
    Распаковывает ZIP и загружает DICOM-серию как 3D объем.
    """
    # Очищаем папку если она была
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Ищем DICOM серию
    reader = sitk.ImageSeriesReader()

    # Рекурсивный поиск папки с DICOM файлами внутри архива
    dicom_names = []
    for root, dirs, files in os.walk(extract_dir):
        names = reader.GetGDCMSeriesFileNames(root)
        if names:
            dicom_names = names
            break

    if not dicom_names:
        raise ValueError("В архиве не найдена валидная DICOM серия.")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Конвертируем в массив (D, H, W)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    return volume
