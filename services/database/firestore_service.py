# firestore_service.py
# Firestore database işlemleri

import os
from datetime import datetime, timezone
import time
from typing import Optional, List, Any
from fastapi import HTTPException
from google.cloud import firestore
import json 
from ..auth.firebase_auth import get_project_id
from services.ml.class_translations import to_tr_label

# Firestore client - lazy initialization için fonksiyon kullanacağız
_fs_client = None

def get_firestore_client():
    """Firestore client'ı lazy initialization ile döndür"""
    global _fs_client
    if _fs_client is None:
        _fs_client = firestore.Client(project=get_project_id())
    return _fs_client


def threads_col(uid: str):
    """Kullanıcının thread koleksiyonunu döndür"""
    return get_firestore_client().collection("users").document(uid).collection("threads")


def thread_ref(uid: str, thread_id: str):
    """Belirli bir thread referansını döndür"""
    return threads_col(uid).document(thread_id)


def messages_col(uid: str, thread_id: str):
    """Thread'in mesaj koleksiyonunu döndür"""
    return thread_ref(uid, thread_id).collection("messages")


def plants_col(uid: str):
    """Kullanıcının bitkiler koleksiyonunu döndür"""
    return get_firestore_client().collection("users").document(uid).collection("plants")


def plant_ref(uid: str, plant_id: str):
    """Belirli bir bitki doküman referansını döndür"""
    return plants_col(uid).document(plant_id)


def update_plant_disease(
    uid: str,
    plant_id: str,
    *,
    cls: str,
    cls_tr: str,
    conf: float,
    thread_id: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> None:
    """Bitkinin hastalık alanını (son teşhis) güncelle.

    Firestore path: users/{uid}/plants/{plant_id}
    Alanlar merge edilir; doküman yoksa oluşturulur.
    """
    if not plant_id:
        return

    now = datetime.now(timezone.utc)
    entry = {
        "class": cls,
        "classTr": cls_tr,
        "confidence": conf,
        "at": now,
        "threadId": thread_id or None,
        "imageRef": image_ref or None,
    }

    ref = plant_ref(uid, plant_id)

    # diseaseHistory: map (keyed) + lastDisease: map
    # Key example: 1734631234567 (ms since epoch) to keep ordering-friendly keys.
    key = str(int(time.time() * 1000))

    snap = ref.get()
    if not snap.exists:
        ref.set(
            {
                "lastDisease": entry,
                "diseaseHistory": {key: entry},
            },
            merge=True,
        )
        return

    # Ensure lastDisease always updated; add one new history entry without overwriting the whole map.
    ref.set({"lastDisease": entry}, merge=True)
    ref.update({f"diseaseHistory.{key}": entry})


def ensure_thread(
    uid: str,
    thread_id: Optional[str],
    *,
    new_thread: bool = False,
    initial_meta: Optional[dict] = None
) -> str:
    """
    Thread'i garanti et:
    - thread_id verilirse: doğrula/yoksa oluştur.
    - new_thread=True ise: her zaman YENİ thread aç.
    - aksi halde: var olan ilk thread'i kullan; yoksa oluştur.
    """
    ALWAYS_NEW_THREAD_ON_INIT = os.getenv("ALWAYS_NEW_THREAD_ON_INIT", "0") == "1"
    col = threads_col(uid)

    # 1) Belirli bir thread istenmişse
    if thread_id:
        ref = col.document(thread_id)
        snap = ref.get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail="Thread not found")
        return thread_id

    # 2) Zorla yeni thread
    if new_thread or ALWAYS_NEW_THREAD_ON_INIT:
        ref = col.document()
        payload = {"createdAt": datetime.now(timezone.utc)}
        if initial_meta: 
            payload.update(initial_meta)
        ref.set(payload)
        return ref.id

    # 3) Mevcut varsa onu kullan, yoksa oluştur
    existing = list(col.limit(1).stream())
    if existing:
        return existing[0].id

    ref = col.document()
    ref.set({"createdAt": datetime.now(timezone.utc)})
    return ref.id


def add_message(uid: str, thread_id: str,
                role: str, content: Any, meta: Optional[dict] = None) -> str:
    """Thread'e mesaj ekle"""
    doc = messages_col(uid, thread_id).document()
    doc.set({
        "role": role,                     # "user" | "assistant" | "systemEvent"
        "content": content,               # user/assistant: string; systemEvent: JSON
        "createdAt": firestore.SERVER_TIMESTAMP,
        "meta": meta or {},
    })
    return doc.id


def update_thread_title(uid: str, thread_id: str, title: str) -> None:
    """Thread'in title'ını güncelle"""
    thread_ref(uid, thread_id).update({"title": title})


def is_first_assistant_message(uid: str, thread_id: str) -> bool:
    """Bu thread'de ilk assistant mesajı mı kontrol et"""
    messages = messages_col(uid, thread_id).where("role", "==", "assistant").limit(1).stream()
    return len(list(messages)) == 0


def update_last_diagnosis(uid: str, thread_id: str,
                          cls: str, conf: float, image_ref: Optional[str] = None):
    """Thread'in son teşhis bilgisini güncelle"""
    tr = to_tr_label(cls)
    thread_ref(uid, thread_id).set({
        "lastDiagnosis": {
            "class": cls,
            "classTr": tr,
            "confidence": conf,
            "at": datetime.now(timezone.utc),
            "imageRef": image_ref or None
        }
    }, merge=True)


def fetch_recent_messages(uid: str, thread_id: str, limit_n: int = 20) -> List[dict]:
    """Thread'den son mesajları getir"""
    q = messages_col(uid, thread_id).order_by(
        "createdAt", direction=firestore.Query.DESCENDING
    ).limit(limit_n)
    docs = list(q.stream())
    items = [d.to_dict() for d in docs]
    items.reverse()
    return items

def _stringify_for_budget(m: dict) -> str:
    r = m.get("role", "")
    c = m.get("content", "")
    if isinstance(c, dict): c = json.dumps(c, ensure_ascii=False)
    return f"{r}:{c}\n"

def trim_history_by_chars(history: List[dict], max_chars: int) -> List[dict]:
    total = 0
    kept = []
    # en sondan başlayıp geriye topla
    for m in reversed(history):
        s = _stringify_for_budget(m)
        if total + len(s) > max_chars:
            break
        kept.append(m)
        total += len(s)
    kept.reverse()
    return kept


def get_thread_memory(uid: str, thread_id: str) -> dict:
    snap = thread_ref(uid, thread_id).get()
    data = snap.to_dict() or {}
    return data.get("memory", {})

def save_thread_memory(uid: str, thread_id: str, memory: dict):
    thread_ref(uid, thread_id).set({"memory": memory}, merge=True)


def get_plant_disease_context(uid: str, plant_id: str, *, limit_n: int = 5) -> dict:
    """Bitkinin hastalık geçmişini LLM'e göndermeye uygun şekilde getir.

    Dönen dict:
      {
        "plantId": str,
        "lastDisease": {...} | None,
        "history": [ {...}, ... ]
      }
    """
    if not plant_id:
        return {"plantId": plant_id, "lastDisease": None, "history": []}

    snap = plant_ref(uid, plant_id).get()
    if not snap.exists:
        return {"plantId": plant_id, "lastDisease": None, "history": []}

    data = snap.to_dict() or {}
    last_disease = data.get("lastDisease")
    history_map = data.get("diseaseHistory") or {}

    entries = []
    if isinstance(history_map, dict):
        for _, v in history_map.items():
            if isinstance(v, dict):
                entries.append(v)

    def _sort_key(e: dict):
        at = e.get("at")
        # Firestore Timestamp -> has datetime() usually; but keep generic
        try:
            if hasattr(at, "datetime"):
                return at.datetime()
        except Exception:
            pass
        return at or datetime.min.replace(tzinfo=timezone.utc)

    entries.sort(key=_sort_key)
    if limit_n and limit_n > 0:
        entries = entries[-limit_n:]

    return {"plantId": plant_id, "lastDisease": last_disease, "history": entries}

