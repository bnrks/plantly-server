# predictService.py - Model ve helper fonksiyonları
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from PIL import Image


def _repo_root() -> Path:
    # services/predictService.py -> repo root
    return Path(__file__).resolve().parents[1]


# ── MODEL + CLASSES yükle
# Yeni inference modeli
MODEL_PATH = _repo_root() / "ml" / "models" / "mobilenetv2_final.keras"

# Model class listesi (model output index -> label)
CLASSES_PATH = _repo_root() / "ml" / "classes" / "classes.json"

try:
    model = keras.models.load_model(str(MODEL_PATH))
except Exception as e:
    hint = ""
    msg = str(e)
    if "keras.src.models.functional" in msg or "batch_shape" in msg:
        hint = (
            "\n\nİPUCU: Bu model dosyası muhtemelen Keras 3 ile kaydedildi. "
            "Conda env içinde `tensorflow>=2.16` (ve beraberinde gelen Keras) kullanın "
            "ya da modeli TF 2.15 uyumlu formatta (SavedModel/H5) yeniden export edin."
        )
    raise RuntimeError(f"Model yüklenemedi ({MODEL_PATH}): {e}{hint}")

try:
    CLASSES = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if not isinstance(CLASSES, list) or not all(isinstance(x, str) for x in CLASSES):
        raise ValueError("classes.json bir string listesi olmalı")
except Exception as e:
    raise RuntimeError(f"Classes yüklenemedi ({CLASSES_PATH}): {e}")


IMG_SIZE = (256, 256)


def preprocess(img: Image.Image) -> np.ndarray:
    """Decode edilmiş PIL image → RGB → 256x256 → float32[0-255] → (1, 256, 256, 3)"""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # 0-255
    return arr[None, ...]