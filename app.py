# app.py ───────────── FastAPI + Keras (CPU/GPU) servis
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io, os, httpx, time, json, pathlib
from dotenv import load_dotenv

# ── Ortam değişkenleri
load_dotenv()

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
    return arr[None, ...]                     # batch=1

# ── FastAPI
app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        x = preprocess(Image.open(io.BytesIO(await file.read())))
        probs = model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        return JSONResponse({
            "class": CLASSES[idx],
            "confidence": float(probs[idx]),
            "probs": [float(p) for p in probs],
            "latency_ms": int((time.time()-t0)*1000)
        })
    except Exception as e:
        raise HTTPException(500, f"Predict hatası: {e}")

# ── Groq Chat  (ham JSON döndürür)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class ChatRequest(BaseModel):
    prompt: str

@app.post("/groq-chat")
async def groq_chat_endpoint(req: ChatRequest):
    try:
        print('groq api key:', GROQ_API_KEY)
        if not GROQ_API_KEY:
            raise HTTPException(500, "GROQ_API_KEY tanımlı değil.")

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            # ---------- ROLE & TASK ----------
                            "You are an expert AI assistant specialized in plant health, care, and identification.You will receive:• a class label (one of: 'bacterial_spot', 'early_blight', 'late_blight', 'healthy') "
                            "• a confidence score (percentage)"
                            "When referring to the disease, **never** output the raw label. Map each label to its Turkish disease name and use that name in your paragraph and suggestions:"
                            "bacterial_spot → Bakteriyel Lekelenme\n"
                            "early_blight → Erken Yanıklık\n"
                            "late_blight → Geç Yanıklık\n"
                            "healthy → Sağlıklı\n"
                            "Keep explanations simple; do not overwhelm the user with technical details.You can say the  disease to user like 'Bitkinizin sahip oldugu hastalik .....'\n\n"

                            # ---------- OUTPUT FORMAT ----------
                            "Always answer the user's question **in Turkish** and return **only** the following valid JSON "
                            "structure — no markdown, code fences, or extra text:\n\n"
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

                            # ---------- STRICT RULES ----------
                            "RULES:\n"
                            "1. Output **must** be valid JSON only; do not add any other characters.\n"
                            "2. The \"paragraph\" field: 2–3 concise sentences in Turkish summarizing the plant's status.\n"
                            "3. The \"suggestions\" array: 3–6 actionable care tips, each as its own string (Turkish).\n"
                            "4. Preserve all key names, quotation marks, brackets, and commas exactly as shown.\n"
                            "5. No trailing commas, missing quotes, or additional formatting—invalid JSON will be rejected.\n"
                        ),
                    },
                    {"role": "user", "content": req.prompt},
                ],
            }


        print("Groq API'ye istek gönderiliyor...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        
        print(f"Groq API response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Groq API error response: {resp.text}")
            raise HTTPException(resp.status_code, f"Groq API hatası: {resp.text}")

        response_json = resp.json()
        print(f"Groq API response: {response_json}")
        
        if "choices" not in response_json or len(response_json["choices"]) == 0:
            raise HTTPException(500, "Groq API'den geçersiz response alındı")
        
        groq_response_content = response_json["choices"][0]["message"]["content"]
        print(f"Groq content: {groq_response_content}")
        
        # Groq'dan gelen JSON string'i doğrula
        try:
            json.loads(groq_response_content)  # Validation için
        except json.JSONDecodeError as je:
            print(f"JSON parse error: {je}")
            raise HTTPException(500, f"Groq'dan geçersiz JSON alındı: {je}")
        
        # Client'ın beklediği format: { "answer": "JSON string" }
        return JSONResponse({
            "answer": groq_response_content
        })
        
    except HTTPException:
        raise  # HTTPException'ları olduğu gibi re-raise et
    except Exception as e:
        print(f"Unexpected error in groq_chat_endpoint: {e}")
        raise HTTPException(500, f"Beklenmeyen hata: {str(e)}")

@app.get("/ping")
def ping():
    return {"msg": "pong"}
