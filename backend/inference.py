import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import os
from model_unet import UNet3D
from dicom_utils import load_dicom_from_zip

MODEL_PATH = "/app/weights/best_model.pth"
PATCH_SIZE = (64, 128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None


def get_model():
    global _model
    if _model is None:
        _model = UNet3D(in_channels=4, out_channels=1, base_features=16)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model


def load_any_format(file_path, temp_extract_dir):
    if file_path.endswith('.zip'):
        return load_dicom_from_zip(file_path, temp_extract_dir)
    else:
        # Для NIfTI SimpleITK возвращает (D, H, W), но nibabel возвращает (H, W, D)
        data = nib.load(file_path).get_fdata(dtype=np.float32)
        return np.transpose(data, (2, 0, 1))  # Приводим к (D, H, W)


def load_and_normalize(file_paths: dict, session_dir: str):
    imgs = []
    affine = np.eye(4)

    for m in ['flair', 't1', 't1ce', 't2']:
        path = file_paths[m]
        extract_dir = os.path.join(session_dir, f"dicom_{m}")
        vol = load_any_format(path, extract_dir)

        if path.endswith(('.nii', '.nii.gz')) and m == 'flair':
            # Оставляем оригинальный affine для NIfTI
            affine = nib.load(path).affine

        imgs.append(vol)

    # Теперь все волюмы гарантированно (D, H, W). Стакаем в (C, D, H, W)
    image = np.stack(imgs, axis=0)
    image_torch = torch.from_numpy(image.copy()).float()

    for c in range(4):
        ch = image_torch[c]
        mask = ch > 0
        if mask.any():
            image_torch[c][mask] = (ch[mask] - ch[mask].mean()) / (ch[mask].std() + 1e-8)

    flair_raw = image[0]
    return image_torch, affine, flair_raw


@torch.no_grad()
def sliding_window_inference(model, image, patch_size=PATCH_SIZE, overlap=0.5):
    c, d, h, w = image.shape
    pd, ph, pw = patch_size
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]

    pad_d, pad_h, pad_w = max(0, pd - d), max(0, ph - h), max(0, pw - w)
    image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
    _, d, h, w = image.shape

    prob_map = torch.zeros((1, d, h, w), device=DEVICE)
    count_map = torch.zeros((1, d, h, w), device=DEVICE)

    z_steps = sorted(set(list(range(0, max(d - pd, 1), stride[0])) + [d - pd]))
    y_steps = sorted(set(list(range(0, max(h - ph, 1), stride[1])) + [h - ph]))
    x_steps = sorted(set(list(range(0, max(w - pw, 1), stride[2])) + [w - pw]))

    for z in z_steps:
        for y in y_steps:
            for x in x_steps:
                patch = image[:, z:z + pd, y:y + ph, x:x + pw].unsqueeze(0).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    logits = model(patch)
                probs = torch.sigmoid(logits)[0]
                prob_map[:, z:z + pd, y:y + ph, x:x + pw] += probs
                count_map[:, z:z + pd, y:y + ph, x:x + pw] += 1

    prob_map = prob_map / count_map
    return prob_map[:, :d - pad_d if pad_d else d,
           :h - pad_h if pad_h else h,
           :w - pad_w if pad_w else w]


def run_inference(file_paths: dict, session_dir: str, threshold=0.5):
    model = get_model()
    image, affine, flair_raw = load_and_normalize(file_paths, session_dir)
    prob_map = sliding_window_inference(model, image)
    mask = (prob_map[0] >= threshold).cpu().numpy().astype(np.uint8)
    return mask, affine, flair_raw
