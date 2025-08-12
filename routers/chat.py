# routers/chat.py - Chat endpoint'leri
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import json
import os

router = APIRouter(tags=["chat"])

# Groq Chat
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class ChatRequest(BaseModel):
    prompt: str

@router.post("/groq-chat")
async def groq_chat_endpoint(req: ChatRequest):
    """Groq Chat endpoint'i"""
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
                            "structure (without any markdown formatting, backticks, or extra text):\n\n"
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
