# routers/predict.py - Predict endpoint'leri
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
from services.predictService import run_cnn_prediction
from services.ml.class_translations import to_tr_label

router = APIRouter(tags=["predict"])

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Bitki hastalığı tahmin endpoint'i"""
    t0 = time.time()
    try:
        # Resim dosyasını oku ve işle
        if hasattr(file, 'read'):
            image_data = await file.read()
        else:
            # Alternatif okuma yöntemi
            image_data = await file.file.read()

        cls, conf, probs = run_cnn_prediction(image_data)
        cls_tr = to_tr_label(cls)
        
        return JSONResponse({
            "class": cls,
            "classTr": cls_tr,
            "confidence": conf,
            "probs": probs,
            "latency_ms": int((time.time()-t0)*1000)
        })
    except Exception as e:
        raise HTTPException(500, f"Predict hatası: {e}")
