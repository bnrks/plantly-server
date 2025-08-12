# app.py ───────────── FastAPI Ana Uygulama
from fastapi import FastAPI
from dotenv import load_dotenv
from routers import predict, chat

# ── Ortam değişkenleri
load_dotenv()

# ── FastAPI
app = FastAPI(
    title="Plantly Server",
    description="Bitki hastalığı tespiti ve AI chat servisi",
    version="1.0.0"
)

# ── Router'ları dahil et
app.include_router(predict.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "Plantly Server is running!"}

@app.get("/ping")
def ping():
    return {"msg": "pong"}
