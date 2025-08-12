# routers/predict.py - Predict endpoint'leri
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import time
from PIL import Image
import io
from services.predictService import preprocess, CLASSES, model

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
            
        img = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        x = preprocess(img)
        
        # Model ile tahmin yap
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        return JSONResponse({
            "class": CLASSES[idx],
            "confidence": float(probs[idx]),
            "probs": [float(p) for p in probs],
            "latency_ms": int((time.time()-t0)*1000)
        })
    except Exception as e:
        raise HTTPException(500, f"Predict hatası: {e}")
