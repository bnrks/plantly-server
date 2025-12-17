# groq_service.py
# Groq AI chat servisi

import os
import json
from typing import List, Optional
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from services.ml.class_translations import to_tr_label

# .env'yi mümkün olduğunca erken yükle (env read'leri doğru olsun)
load_dotenv()

# === Context & Memory Ayarları ===
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "8000"))   # LLM bağlam bütçesi ~8k karakter
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "1") == "1"
MEMORY_REFRESH_EVERY = int(os.getenv("MEMORY_REFRESH_EVERY", "3"))  # her 3 mesajda bir running summary güncelle
MEM_FACTS_LIMIT = int(os.getenv("MEM_FACTS_LIMIT", "8"))            # sabit gerçek sayısı

from ..database.firestore_service import (
    fetch_recent_messages, thread_ref,
    get_thread_memory, save_thread_memory, trim_history_by_chars
)

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    "Sen bitki sağlığı ve bakımında uzman bir asistansın.\n"
    "- Modelden/servisten gelen teşhis etiketleri İngilizce ve teknik olabilir (örn. Apple__Apple_Scab). "
    "Bu ham etiketleri KESİNLİKLE kullanıcıya aynen gösterme.\n"
    "- Teşhis varsa bunu mutlaka Türkçe ve sade şekilde ifade et (örn. 'Elma - Karalekesi').\n"
    "- Kullanıcıyı boğmadan net, adım adım öneriler ver.\n"
    "- Emin değilsen olasılıklardan bahset ve basit kontrol adımları öner.\n"
    "- Her zaman Türkçe yanıtla.\n"
    "- Eğer son kullanıcı mesajı sadece selamlaşma/teşekkür gibi küçük konuşmaysa KISA bir selamlama yap; teşhis veya bakım listesi verme. Gerekirse 'Size nasıl yardımcı olabilirim?' diye sor.\n"
    "- Kullanıcı bakım/hastalıkla ilgili soru sorarsa, varsa son teşhisi de dikkate alarak yanıtla.\n\n"
    "ÖNEMLİ: Yanıtını şu JSON formatında ver:\n"
    "{\n"
    '  "content": "Ana cevabın buraya gelsin",\n'
    '  "notes": ["Sadece gerçek bakım önerileri varsa ekle"]\n'
    "}\n"
    "KURALLER:\n"
    "- content: Her zaman doldur\n"
    "- notes: Sadece bitki bakımı/hastalık tedavisi önerileri varsa doldur\n"
    "- Selamlaşma, teşekkür, genel sohbet durumlarında notes boş array [] ver\n"
    "- notes her elemanı kısa ve net olsun (maksimum 1 cümle)\n"
    "- notes sadece actionable (yapılabilir) öneriler içersin"
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


def build_llm_messages(uid: str, thread_id: str, user_text: Optional[str],
                       *, append_user_text: bool = True,
                       force_include_diag: Optional[bool] = None,
                       history_k: int = 20) -> List[dict]:

    t_snap = thread_ref(uid, thread_id).get()
    t_data = t_snap.to_dict() or {}
    last_diag = t_data.get("lastDiagnosis")
    memory = (t_data.get("memory") or {}) if MEMORY_ENABLED else {}

    history = fetch_recent_messages(uid, thread_id, limit_n=history_k)
    # (A) bütçeye göre kırp
    history = trim_history_by_chars(history, HISTORY_MAX_CHARS)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # (B) hafıza (özet + sabit gerçekler) en başa
    if memory:
        if memory.get("summary"):
            messages.append({"role":"system", "content": f"Konuşma özeti: {memory['summary']}"})
        facts = memory.get("facts") or []
        if facts:
            messages.append({"role":"system", "content": "Sabit gerçekler:\n- " + "\n- ".join(facts)})

    # teşhis ekleme logic'i (senin mevcut kodun)
    include_diag = False
    if force_include_diag is not None:
        include_diag = force_include_diag
    else:
        include_diag = not is_smalltalk(user_text)

    if last_diag and include_diag:
        tr_name = (last_diag.get("classTr") or "").strip() or to_tr_label(str(last_diag.get("class") or ""))
        diag_line = f"Son teşhis: {tr_name} (%{round(float(last_diag.get('confidence', 0))*100)})"
        messages.append({"role": "system", "content": diag_line})

    # geçmiş mesajlar (senin mevcut döngün)
    for m in history:
        r = m.get("role")
        c = m.get("content", "")
        if r == "systemEvent":
            try:
                payload = c if isinstance(c, dict) else json.loads(c)
                if payload.get("type") == "diagnosis" and include_diag:
                    cls = payload.get("class")
                    tr = to_tr_label(str(cls or ""))
                    conf = float(payload.get("confidence", 0))
                    messages.append({"role": "system", "content": f"Teşhis: {tr} (%{round(conf*100)})"})
            except Exception:
                pass
        elif r in ("user", "assistant"):
            if isinstance(c, dict):
                c = json.dumps(c, ensure_ascii=False)
            messages.append({"role": r, "content": c})

    if user_text and append_user_text:
        messages.append({"role": "user", "content": user_text})

    return messages


async def call_groq_api(messages: List[dict]) -> str:
    """Groq API'ye çağrı yap - sadece raw text döndürür"""
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


async def call_groq_api_structured(messages: List[dict]) -> dict:
    """Groq API'ye çağrı yap ve JSON yanıt parse et"""
    raw_response = await call_groq_api(messages)
    
    try:
        # JSON parse et
        response_data = json.loads(raw_response)
        
        # Doğrula
        if not isinstance(response_data, dict):
            raise ValueError("Response dict değil")
            
        content = response_data.get("content", "")
        notes = response_data.get("notes", [])
        
        if not isinstance(notes, list):
            notes = []
            
        return {
            "content": content,
            "notes": notes
        }
        
    except (json.JSONDecodeError, ValueError, KeyError):
        # JSON parse edilemezse fallback
        return {
            "content": raw_response,
            "notes": []
        }


async def generate_conversation_title(user_message: str) -> str:
    """İlk kullanıcı mesajına göre konuşma başlığı üret"""
    title_prompt = f"""Aşağıdaki kullanıcı mesajına göre kısa ve öz bir başlık oluştur. 
Başlık maksimum 5-6 kelime olsun ve bitki sağlığı/hastalıkları konusuyla ilgili olsun.

Kullanıcı mesajı: "{user_message}"

Sadece başlığı döndür, başka bir şey yazma."""

    messages = [{"role": "user", "content": title_prompt}]
    
    try:
        title = await call_groq_api(messages)
        # Başlığı temizle ve kısalt
        title = title.strip().strip('"').strip("'")
        if len(title) > 60:
            title = title[:60] + "..."
        return title
    except Exception:
        # Hata durumunda varsayılan başlık
        return "Bitki Sağlığı Danışmanlığı"


def generate_fallback_reply(cls: str, conf: float) -> str:
    """Teşhis için yedek yanıt oluştur"""
    tr = to_tr_label(cls)
    pct = round(conf*100)

    # label format: Plant__Condition
    condition = (cls.split("__", 1)[1] if "__" in (cls or "") else "").lower()

    if condition == "healthy":
        tips = [
            "Aşırı sulamadan kaçın ve saksı drenajını koru.",
            "Haftada 1–2 kez genel durum kontrolü yap.",
            "Güneş ve hava sirkülasyonunu yeterli tut."
        ]
    elif "bacterial_spot" in condition:
        tips = [
            "Etkilenen yaprakları steril makasla uzaklaştır.",
            "Yaprakları ıslatmadan dipten sulama yap.",
            "Bitkiler arasında hava sirkülasyonu için mesafe bırak.",
            "Bakır içerikli ürünleri etiketine uygun ve gerektiğinde kullanmayı değerlendir."
        ]
    elif "early_blight" in condition:
        tips = [
            "Hasta yaprakları topla ve çöpe at (kompost yapma).",
            "Sulamayı sabah erken saatlerde ve toprağa yap.",
            "Alt yaprakları seyreltip hava akışını artır.",
            "Gerekirse etiketine uygun mantar hastalığına yönelik ürün kullan."
        ]
    else:  # diğer hastalıklar
        if "late_blight" in condition:
            tips = [
                "Şiddetli lekeli yaprakları derhal uzaklaştır.",
                "Yaprak ıslaklığını azalt: üstten sulamadan kaçın.",
                "Bitkiyi iyi havalanan bir konuma al.",
                "Gerekirse uygun fungisitleri etiketine uygun kullanmayı değerlendir."
            ]
        elif "apple_scab" in condition:
            tips = [
                "Dökülen yaprakları/lekeli yaprakları topla ve imha et.",
                "Üstten sulamadan kaçın; yaprak ıslaklığını azalt.",
                "Ağacın içini havalandıracak şekilde budama yapmayı değerlendir.",
                "Gerekirse etiketine uygun mantar hastalığına yönelik koruyucu uygulamaları düşün."
            ]
        elif "black_rot" in condition:
            tips = [
                "Çürüyen meyveleri ve lekeli yaprakları ortamdan uzaklaştır.",
                "Budama artıkları ve döküntüleri bahçede bırakma.",
                "Bitkiyi fazla sıkıştırma; hava sirkülasyonunu artır.",
                "Gerekirse etiketine uygun fungisitleri değerlendirebilirsin."
            ]
        elif "cedar_apple_rust" in condition:
            tips = [
                "Enfekte yaprakları temizle; döküntüleri toplayıp imha et.",
                "Hava sirkülasyonunu artır; yaprakların hızlı kurumasını sağla.",
                "Yakında ardıç/servi türleri varsa kaynak olabileceğini unutma.",
                "Gerekirse etiketine uygun pas hastalığına yönelik ürünleri değerlendir."
            ]
        elif "powdery_mildew" in condition:
            tips = [
                "Hasta yaprakları temizle; bitkiler arası hava akışını artır.",
                "Üstten sulamadan kaçın; yaprakları kuru tut.",
                "Gerekirse etiketine uygun külleme ilacı kullanmayı değerlendir."
            ]
        elif "rust" in condition:
            tips = [
                "Enfekte yaprakları temizle ve çevreye saçılmasını önle.",
                "Hava sirkülasyonunu artır; bitkiyi çok sık dikme.",
                "Gerekirse etiketine uygun pas hastalığına yönelik ürün kullanmayı değerlendir."
            ]
        elif "gray_leaf_spot" in condition:
            tips = [
                "Alt ve çok lekeli yaprakları temizle; bitki sıklığını azalt.",
                "Üstten sulamayı azalt; yaprakların hızlı kurumasını sağla.",
                "Tarlada/alan içinde bitki artıkları yönetimine dikkat et.",
                "Gerekirse etiketine uygun fungisitleri değerlendirebilirsin."
            ]
        elif "northern_leaf_blight" in condition:
            tips = [
                "Şiddetli etkilenen yaprakları temizle ve imha et.",
                "Bitkiler arasında hava akışını artır.",
                "Üstten sulamadan kaçın; yaprak ıslaklığını azalt.",
                "Gerekirse etiketine uygun mantar hastalığına yönelik ürün kullanmayı değerlendir."
            ]
        elif "esca" in condition:
            tips = [
                "Şiddetli etkilenen sürgün/omcaları budama ile ayırmayı değerlendir.",
                "Budama aletlerini dezenfekte et; bulaş riskini azalt.",
                "Bitki stresini azalt: düzenli sulama ve dengeli gübreleme.",
                "Belirtiler yaygınsa bağ uzmanı/zirai danışmandan destek al."
            ]
        elif "leaf_blight" in condition:
            tips = [
                "Lekeli yaprakları temizle ve imha et.",
                "Yaprak ıslaklığını azalt; toprağa/dipten sulamayı tercih et.",
                "Hava sirkülasyonunu artır; bitkiyi sıkıştırma.",
                "Gerekirse etiketine uygun fungisitleri değerlendirebilirsin."
            ]
        elif "leaf_scorch" in condition:
            tips = [
                "Kuruyan/yanık görünümlü yaprakları temizle.",
                "Sulama düzenini kontrol et; toprak tamamen kurumadan sulamayı planla.",
                "Sıcak/kurak günlerde doğrudan öğle güneşini azaltmayı değerlendir.",
                "Belirtiler hızla artarsa hastalık olasılığı için ek görsel/uzman görüşü al."
            ]
        else:
            tips = [
                "Hastalıklı görünen yaprakları temizle ve at.",
                "Üstten sulamadan kaçın; toprağa/dipten sulamayı tercih et.",
                "Bitkiyi iyi havalanan bir konuma al ve yoğunluğu azalt.",
                "Belirtiler artarsa yerel bir ziraat bayii/uzmanla görüş."
            ]
    
    para = f"Son teşhise göre **{tr}** olasılığı yüksek (≈%{pct}). Aşağıdaki adımları uygulayabilirsin:"
    return para + "\n- " + "\n- ".join(tips)


async def summarize_into_memory(uid: str, thread_id: str, recent: List[dict]):
    """
    recent: bu turda eklenecek kısa geçmiş (trimlenmiş)
    memory: summary + facts güncellenir
    """
    if not MEMORY_ENABLED:
        return

    prev = get_thread_memory(uid, thread_id)
    prev_summary = prev.get("summary", "")
    prev_facts = prev.get("facts", [])

    # LLM’e temiz, kısa bir özet isteyeceğiz (JSON zorunlu)
    sys = (
        "Aşağıdaki sohbet dökümünden KISA bir özet çıkar ve kalıcı, değişmesi zor 'sabit gerçekleri' listele. "
        "Sadece JSON döndür:\n"
        "{ \"summary\": \"1-2 cümle\", \"facts\": [\"...\", \"...\"] }"
    )

    msgs = [{"role":"system","content":sys}]
    if prev_summary or prev_facts:
        msgs.append({"role":"system","content":f"Önceki özet: {prev_summary}\nÖnceki sabitler: {prev_facts}"})

    # son 8-12 ile sınırlı küçük bir blok yeterli
    block = recent[-12:]
    for m in block:
        r = m.get("role")
        c = m.get("content")
        if isinstance(c, dict): c = json.dumps(c, ensure_ascii=False)
        # systemEvent teşhisleri ipucu olsun
        if r == "systemEvent":
            try:
                p = m["content"]
                if isinstance(p, str): p = json.loads(p)
                if p.get("type") == "diagnosis":
                    tr = to_tr_label(str(p.get("class") or ""))
                    c = f"[TEŞHİS] {tr} (%{round(float(p.get('confidence',0))*100)})"
            except Exception:
                pass
            r = "system"
        msgs.append({"role": r, "content": c})

    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": GROQ_MODEL, "messages": msgs, "temperature": 0.2}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        data = resp.json()
        out = (data.get("choices",[{}])[0].get("message",{}) or {}).get("content","").strip()
        mem = json.loads(out)
        # kısıtla ve birleştir
        facts = (mem.get("facts") or [])[:MEM_FACTS_LIMIT]
        summary = (mem.get("summary") or prev_summary).strip()
        memory = {
            "summary": summary if summary else prev_summary,
            "facts": facts if facts else prev_facts,
            "updatedAt": datetime.now(timezone.utc),
            "msgCount": int(prev.get("msgCount",0)) + 1
        }
        save_thread_memory(uid, thread_id, memory)
    except Exception:
        # sessiz fallback: en azından sayaç güncellensin
        memory = {
            "summary": prev_summary,
            "facts": prev_facts,
            "updatedAt": datetime.now(timezone.utc),
            "msgCount": int(prev.get("msgCount",0)) + 1
        }
        save_thread_memory(uid, thread_id, memory)
