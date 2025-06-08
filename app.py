from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import os
import httpx
from dotenv import load_dotenv
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import time
# .env dosyasını yükle
load_dotenv()

# --- TFLite Model Yükleme ve Fonksiyonlar ---
MODEL_PATH = "model.tflite"  # Gerekirse yolu güncelleyin
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
outp = interpreter.get_output_details()[0]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((256, 256)).convert("RGB")
    arr = (np.asarray(img, np.float32) / 255.0)[None, ...]
    return arr

def predict(arr: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(inp["index"], arr)
    interpreter.invoke()
    return interpreter.get_tensor(outp["index"])

# --- Groq API Konfigürasyonu ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Pydantic Modelleri
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

# FastAPI Uygulaması
app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    start = time.time()
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        preds = predict(preprocess(img))[0].tolist()
        labels = ["healthy", "powdery", "rust"]
        print("Predict tamamlandı, süre:", time.time() - start)
        return JSONResponse({
            "class": labels[int(np.argmax(preds))],
            "probs": preds
        })
    except Exception as e:
        print("Predict endpoint hatası:", e)
        raise HTTPException(status_code=500, detail=f"Predict hatası: {e}")
@app.post("/groq-chat", response_model=ChatResponse)
async def groq_chat_endpoint(req: ChatRequest):
    # Anahtar var mı kontrol et
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY bulunamadı veya yüklenmedi.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful plant care assistant that provides advice "
                    "on plant health, care, and identification. Answer the user's question "
                    "in Turkish and return **only** the following JSON structure:\n\n"
                    "{\n"
                    "  \"results\": {\n"
                    "    \"paragraph\": \"<2–3 sentence introductory paragraph>\",\n"
                    "    \"suggestions\": [\n"
                    "      \"<Suggestion 1>\",\n"
                    "      \"<Suggestion 2>\",\n"
                    "      \"<Suggestion 3>\",\n"
                    "      \"...\"\n"
                    "    ]\n"
                    "  }\n"
                    "}\n\n"
                    "- The \"paragraph\" field should contain the short intro.\n"
                    "- The \"suggestions\" array should list each care tip as a separate string."
                    "* \"suggestions\" must be a JSON array; **each** care tip is one element in that array."
                ),
            },
            {"role": "user", "content": (req.prompt)},
        ],
    }

    async with httpx.AsyncClient() as client:
        # İstek gönderme hatası
        try:
            resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Groq API isteği başarısız: {e}")

        # HTTP durum kodunu kontrol et
        if resp.status_code != 200:
            err_text = resp.text
            raise HTTPException(status_code=resp.status_code, detail=f"Groq API hatası: {err_text}")

        # JSON işleme ve cevabı alma
        try:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Groq cevabı işlenirken hata: {e}")

    return ChatResponse(answer=answer)


@app.get("/ping")
def ping():
    return {"msg": "pong"}
