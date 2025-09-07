# ws_chat.py
# FastAPI WebSocket + Firestore sohbet (threads under users/{uid}/threads), plantId ZORUNLU DEĞİL

import os
import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Header

# Local imports
from services.connection.websocket_manager import WebSocketManager
from services.auth.firebase_auth import verify_id_token_or_raise
from services.database.firestore_service import (
    ensure_thread, add_message, update_last_diagnosis, fetch_recent_messages,
    update_thread_title, is_first_assistant_message
)
from services.chat.groq_service import (
    build_llm_messages, call_groq_api, call_groq_api_structured, generate_fallback_reply, summarize_into_memory,
    generate_conversation_title
)
from services.ml.prediction_service import run_cnn_prediction

# -------------------- .env & Configuration --------------------
from dotenv import load_dotenv
load_dotenv()

ALWAYS_NEW_THREAD_ON_INIT = os.getenv("ALWAYS_NEW_THREAD_ON_INIT", "0") == "1"
# === Context & Memory Ayarları ===
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "8000"))   # LLM bağlam bütçesi ~8k karakter
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "1") == "1"
MEMORY_REFRESH_EVERY = int(os.getenv("MEMORY_REFRESH_EVERY", "3"))  # her 3 mesajda bir running summary güncelle
MEM_FACTS_LIMIT = int(os.getenv("MEM_FACTS_LIMIT", "8"))            # sabit gerçek sayısı

router = APIRouter()

# -------------------- Connection Manager --------------------
manager = WebSocketManager()


# -------------------- WebSocket Endpoint --------------------
@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    """
    İlk mesaj (init):
    {
      "type": "init",
      "idToken": "<firebase id token>",
      "thread_id": "opsiyonel"
    }

    Sonraki mesaj tipleri:
    - {"type":"user_text", "text":"Toprak değişmeli mi?"}
    - {"type":"diagnosis", "class":"late_blight", "confidence":0.82, "image_ref":"...", "auto_reply": true}
    - {"type":"ping"}
    """
    await websocket.accept()

    uid = None
    thread_id = None

    try:
        init_msg = await websocket.receive_text()
        init = json.loads(init_msg)
        new_thread_flag = bool(init.get("new_thread")) or ALWAYS_NEW_THREAD_ON_INIT
        title = init.get("title")
        initial_meta = {"title": title} if title else None

        if init.get("type") != "init":
            await websocket.close(code=1002)
            return

        id_token = init.get("idToken")
        if not id_token:
            await websocket.close(code=4401)
            return
        uid = verify_id_token_or_raise(id_token)

        # Thread'i users/{uid}/threads altında oluştur/garantile
        thread_id = ensure_thread(
            uid,
            init.get("thread_id"),
            new_thread=new_thread_flag,
            initial_meta=initial_meta
        )

        # Odaya ekle ve hazır bilgisi
        await manager.connect(thread_id, websocket)
        await websocket.send_text(json.dumps({
            "type": "thread_ready",
            "thread_id": thread_id
        }))

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            mtype = data.get("type")

            if mtype == "user_text":
                text = (data.get("text") or "").strip()
                if not text:
                    continue

                add_message(uid, thread_id, role="user", content=text)

                # İlk assistant mesajı mı kontrol et
                is_first = is_first_assistant_message(uid, thread_id)

                messages = build_llm_messages(uid, thread_id, user_text=None)
                assistant_response = await call_groq_api_structured(messages)

                # İlk mesajsa title üret
                title = None
                if is_first:
                    title = await generate_conversation_title(text)
                    update_thread_title(uid, thread_id, title)

                # Structured response'u database'e kaydet
                asst_mid = add_message(uid, thread_id, role="assistant", content=assistant_response)
                if MEMORY_ENABLED:
                    recent = fetch_recent_messages(uid, thread_id, limit_n=20)
                    if len(recent) % MEMORY_REFRESH_EVERY == 0:
                         await summarize_into_memory(uid, thread_id, recent)
                
                # Response hazırla
                response = {
                    "type": "message",
                    "thread_id": thread_id,
                    "message": {
                        "role": "assistant", 
                        "content": assistant_response["content"],
                        "notes": assistant_response["notes"],
                        "id": asst_mid
                    }
                }
                
                # İlk mesajsa title ekle
                if title:
                    response["title"] = title
                    
                await manager.broadcast(thread_id, response)

            elif mtype == "diagnosis":
                cls = data.get("class")
                conf = float(data.get("confidence", 0))
                image_ref = data.get("image_ref")

                diag_payload = {"type": "diagnosis", "class": cls, "confidence": conf, "imageRef": image_ref}
                diag_mid = add_message(uid, thread_id, role="systemEvent", content=diag_payload)

                update_last_diagnosis(uid, thread_id, cls=cls, conf=conf, image_ref=image_ref)

                await manager.broadcast(thread_id, {
                    "type": "message",
                    "thread_id": thread_id,
                    "message": {"role": "systemEvent", "content": diag_payload, "id": diag_mid}
                })

                if data.get("auto_reply"):
                    messages = build_llm_messages(uid, thread_id, user_text=None)
                    assistant_text = await call_groq_api(messages)
                    asst_mid = add_message(uid, thread_id, role="assistant", content=assistant_text)
                    await manager.broadcast(thread_id, {
                        "type": "message",
                        "thread_id": thread_id,
                        "message": {"role": "assistant", "content": assistant_text, "id": asst_mid}
                    })

            elif mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "error": "Unknown message type"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass
    finally:
        if thread_id:
            manager.disconnect(thread_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/chat/analyze-image")
async def analyze_image(
    id_token: str = Header(..., alias="idToken"),  # Firebase ID token (header)
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    auto_reply: Optional[bool] = Form(False),
):
    """
    Form-Data:
      - file: (required) görüntü dosyası
      - thread_id: (optional) yoksa users/{uid}/threads altında mevcut ilk thread kullanılır, yoksa yeni açılır
      - auto_reply: (optional) True ise teşhisten sonra LLM yanıtı üretir
    Header:
      - idToken: Firebase ID token
    """
    # 1) Auth & thread
    uid = verify_id_token_or_raise(id_token)
    t_id = ensure_thread(uid, thread_id)

    # 2) Dosyayı oku
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Boş dosya")

    # 3) CNN tahmini (aynı process)
    try:
        cls, conf, probs = run_cnn_prediction(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference hatası: {e}")

    # 4) Sohbete systemEvent olarak ekle + lastDiagnosis güncelle
    diag_payload = {"type": "diagnosis", "class": cls, "confidence": conf, "imageRef": None}
    mid = add_message(uid, t_id, role="systemEvent", content=diag_payload)
    update_last_diagnosis(uid, t_id, cls=cls, conf=conf, image_ref=None)

    # 5) WS yayını (açık oda varsa)
    await manager.broadcast(t_id, {
        "type": "message",
        "thread_id": t_id,
        "message": {"role": "systemEvent", "content": diag_payload, "id": mid}
    })

    # 6) (opsiyonel) hemen LLM cevabı üret
    asst = None
    if auto_reply:
        # SANAL USER MESAJI: modelin konuşmayı başlatması için
        auto_user = (
            "Yeni teşhise göre kısa bir değerlendirme yap; 2–3 cümlede durumu özetle ve "
            "4 maddelik uygulanabilir bakım önerisi ver."
        )
        messages = build_llm_messages(
            uid, t_id, user_text=auto_user,
            append_user_text=True,
            force_include_diag=True            # foto sonrası proaktif
        )
        asst_text = await call_groq_api(messages)

        # Boş geldiyse güvenli fallback yaz
        if not asst_text.strip():
            asst_text = generate_fallback_reply(cls, conf)

        asst_mid = add_message(uid, t_id, role="assistant", content=asst_text)
        await manager.broadcast(t_id, {
            "type": "message",
            "thread_id": t_id,
            "message": {"role": "assistant", "content": asst_text, "id": asst_mid}
        })
        asst = {"message_id": asst_mid, "content": asst_text}
    
    return {
        "thread_id": t_id,
        "diagnosis": {"class": cls, "confidence": conf, "probs": probs},
        "message_id": mid,
        "assistant": asst  # auto_reply=false ise null/None; true ise mesaj içeriği var
    }
