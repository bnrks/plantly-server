# prediction_service.py
# ML model tahmin servisi

import numpy as np
from PIL import Image
import io

# predictService'i doğru path'den import et
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.predictService import preprocess, CLASSES, model


def softmax_logits(arr: np.ndarray) -> np.ndarray:
    """Softmax uygula (model çıktısı zaten softmax ise gerek yok; güvence olsun)"""
    e = np.exp(arr - np.max(arr))
    return e / np.sum(e)


def run_cnn_prediction(image_bytes: bytes) -> tuple[str, float, list[float]]:
    """
    Görüntü bytes'ını alır, CNN model ile tahmin yapıp
    (class, confidence, probs[]) döndürür.
    """
    try:
        # Görüntüyü yükle ve ön işleme tabi tut
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(img)  # (1, 256, 256, 3) float32
        
        # Model tahmini
        probs = model.predict(x, verbose=0)[0]  # np.ndarray
        
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs, dtype=np.float32)
        
        # Eğer model output'u softmax değilse normalleştir:
        if np.any(probs < 0) or np.sum(probs) > 1.01:
            probs = softmax_logits(probs)
        
        # En yüksek olasılıklı sınıfı bul
        idx = int(np.argmax(probs))
        cls = CLASSES[idx]
        conf = float(probs[idx])
        
        return cls, conf, [float(p) for p in probs]
        
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {str(e)}")


def get_class_names() -> list:
    """Mevcut sınıf isimlerini döndür"""
    return CLASSES.copy()


def get_class_turkish_names() -> dict:
    """Sınıf isimlerinin Türkçe karşılıklarını döndür"""
    return {
        "healthy": "Sağlıklı",
        "bacterial_spot": "Bakteriyel leke",
        "early_blight": "Erken yanıklık",
        "late_blight": "Geç yanıklık",
    }
