# groq_service.py
# Groq AI chat servisi

import os
import json
from typing import List, Optional
import httpx
from dotenv import load_dotenv

from ..database.firestore_service import get_thread_data, fetch_recent_messages

load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Constants
CLASS_TR = {
    "healthy": "Sağlıklı",
    "bacterial_spot": "Bakteriyel leke",
    "early_blight": "Erken yanıklık",
    "late_blight": "Geç yanıklık",
}

SYSTEM_PROMPT = (
    "Sen bitki sağlığı ve bakımında uzman bir asistansın.\n"
    "- Hastalık isimlerini Türkçe kullan: healthy=Sağlıklı, bacterial_spot=Bakteriyel leke, "
    "early_blight=Erken yanıklık, late_blight=Geç yanıklık.\n"
    "- Kullanıcıyı boğmadan net, adım adım öneriler ver.\n"
    "- Emin değilsen olasılıklardan bahset ve basit kontrol adımları öner.\n"
    "- Her zaman Türkçe yanıtla.\n"
    "- Eğer son kullanıcı mesajı sadece selamlaşma/teşekkür gibi küçük konuşmaysa KISA bir selamlama yap; teşhis veya bakım listesi verme. Gerekirse 'Size nasıl yardımcı olabilirim?' diye sor.\n"
    "- Kullanıcı bakım/hastalıkla ilgili soru sorarsa, varsa son teşhisi de dikkate alarak yanıtla."
)

SMALLTALK_WORDS = {
    "merhaba", "selam", "selamlar", "günaydın", "gunaydin",
    "iyi akşamlar", "iyi aksamlar", "iyi geceler", "iyi günler", "iyi gunler",
    "hello", "hi", "hey", "naber", "nabersin", "nasılsın", "nasilsin",
    "teşekkürler", "tesekkurler", "sağol", "sagol", "thanks", "ok", "tamam"
}


def is_smalltalk(text: Optional[str]) -> bool:
    """Küçük sohbet mesajı olup olmadığını kontrol et"""
    if not text:
        return False
    t = text.lower().strip()
    # kısa ve selam kelimelerinden biri geçiyorsa
    return (len(t) <= 40) and any(w in t for w in SMALLTALK_WORDS)


def build_llm_messages(
    uid: str,
    thread_id: str,
    user_text: Optional[str],
    *,
    append_user_text: bool = True,
    force_include_diag: Optional[bool] = None,
    history_k: int = 20
) -> List[dict]:
    """LLM için mesaj listesi oluştur"""
    t_data = get_thread_data(uid, thread_id)
    last_diag = t_data.get("lastDiagnosis")

    history = fetch_recent_messages(uid, thread_id, limit_n=history_k)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Teşhisi duruma göre ekle
    include_diag = False
    if force_include_diag is not None:
        include_diag = force_include_diag
    else:
        include_diag = not is_smalltalk(user_text)

    if last_diag and include_diag:
        diag_line = f"Son teşhis: {last_diag.get('classTr', last_diag.get('class'))} (%{round(float(last_diag.get('confidence', 0))*100)})"
        messages.append({"role": "system", "content": diag_line})

    # Geçmişi ekle
    for m in history:
        r = m.get("role")
        if r == "systemEvent":
            try:
                payload = m.get("content", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if payload.get("type") == "diagnosis" and include_diag:
                    cls = payload.get("class")
                    tr = CLASS_TR.get(cls, cls)
                    conf = float(payload.get("confidence", 0))
                    messages.append({"role": "system", "content": f"Teşhis: {tr} (%{round(conf*100)})"})
            except Exception:
                pass
        elif r in ("user", "assistant"):
            content = m.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            messages.append({"role": r, "content": content})

    if user_text and append_user_text:
        messages.append({"role": "user", "content": user_text})

    return messages


async def call_groq_api(messages: List[dict]) -> str:
    """Groq API'ye çağrı yap"""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY tanımlı değil.")
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.4}
    
    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
    
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API hatası: {resp.status_code} {resp.text}")
    
    data = resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return (text or "").strip()


def generate_fallback_reply(cls: str, conf: float) -> str:
    """Teşhis için yedek yanıt oluştur"""
    tr_map = {
        "healthy": "Sağlıklı",
        "bacterial_spot": "Bakteriyel leke",
        "early_blight": "Erken yanıklık",
        "late_blight": "Geç yanıklık",
    }
    tr = tr_map.get(cls, cls)
    pct = round(conf*100)
    
    if cls == "healthy":
        tips = [
            "Aşırı sulamadan kaçın ve saksı drenajını koru.",
            "Haftada 1–2 kez genel durum kontrolü yap.",
            "Güneş ve hava sirkülasyonunu yeterli tut."
        ]
    elif cls == "bacterial_spot":
        tips = [
            "Etkilenen yaprakları steril makasla uzaklaştır.",
            "Yaprakları ıslatmadan dipten sulama yap.",
            "Bitkiler arasında hava sirkülasyonu için mesafe bırak.",
            "Bakır içerikli ürünleri etiketine uygun ve gerektiğinde kullanmayı değerlendir."
        ]
    elif cls == "early_blight":
        tips = [
            "Hasta yaprakları topla ve çöpe at (kompost yapma).",
            "Sulamayı sabah erken saatlerde ve toprağa yap.",
            "Alt yaprakları seyreltip hava akışını artır.",
            "Gerekirse etiketine uygun mantar hastalığına yönelik ürün kullan."
        ]
    else:  # late_blight
        tips = [
            "Şiddetli lekeli yaprakları derhal uzaklaştır.",
            "Yaprak ıslaklığını azalt: üstten sulamadan kaçın.",
            "Bitkiyi iyi havalanan bir konuma al.",
            "Gerekirse uygun fungisitleri etiketine uygun kullanmayı değerlendir."
        ]
    
    para = f"Son teşhise göre **{tr}** olasılığı yüksek (≈%{pct}). Aşağıdaki adımları uygulayabilirsin:"
    return para + "\n- " + "\n- ".join(tips)
