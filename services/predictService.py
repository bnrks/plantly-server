# predictService.py - Model ve helper fonksiyonları
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json, pathlib

# ── MODEL + LABELS yükle
MODEL_PATH = "models/plant_disease_classifier_256.keras"
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model yüklenemedi: {e}")

# labels.json varsa oku, yoksa elle sırala (alfabetik)
labels_path = pathlib.Path(MODEL_PATH).with_name("labels.json")
if labels_path.exists():
    CLASSES = json.loads(labels_path.read_text())
else:
    CLASSES = ["bacterial_spot", "early_blight", "healthy", "late_blight"]

IMG_SIZE = (256, 256)

def preprocess(img: Image.Image) -> np.ndarray:
    """PIL → float32[0-255] → (1, 256, 256, 3)"""
    img = img.resize(IMG_SIZE).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)   # 0-255
    return arr[None, ...]