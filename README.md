# 3D Brain Tumor Segmentation MVP

MVP-система для сегментации опухоли мозга по MRI-данным и визуализации результата в 3D.

## Возможности

- Загрузка 4 MRI-модальностей:
  - FLAIR
  - T1
  - T1ce
  - T2
- 3D U-Net сегментация опухоли
- Построение 3D-модели мозга и опухоли
- Веб-визуализация через Three.js
- Контейнеризация через Docker Compose
- Поддержка NIfTI
- Подготовка к поддержке DICOM ZIP-серий

## Структура проекта

```text
backend/
  main.py
  inference.py
  dicom_utils.py
  mesh_utils.py
  model_unet.py
  requirements.txt
  Dockerfile

frontend/
  index.html
  nginx.conf
  Dockerfile

weights/
  best_model.pth

docker-compose.yml
