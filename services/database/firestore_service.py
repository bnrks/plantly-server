# firestore_service.py
# Firestore database işlemleri

import os
from datetime import datetime, timezone
from typing import Optional, List, Any
from fastapi import HTTPException
from google.cloud import firestore

from ..auth.firebase_auth import get_project_id

# Firestore client - lazy initialization için fonksiyon kullanacağız
_fs_client = None

def get_firestore_client():
    """Firestore client'ı lazy initialization ile döndür"""
    global _fs_client
    if _fs_client is None:
        _fs_client = firestore.Client(project=get_project_id())
    return _fs_client

# Class mapping for Turkish translations
CLASS_TR = {
    "healthy": "Sağlıklı",
    "bacterial_spot": "Bakteriyel leke",
    "early_blight": "Erken yanıklık",
    "late_blight": "Geç yanıklık",
}


def threads_col(uid: str):
    """Kullanıcının thread koleksiyonunu döndür"""
    return get_firestore_client().collection("users").document(uid).collection("threads")


def thread_ref(uid: str, thread_id: str):
    """Belirli bir thread referansını döndür"""
    return threads_col(uid).document(thread_id)


def messages_col(uid: str, thread_id: str):
    """Thread'in mesaj koleksiyonunu döndür"""
    return thread_ref(uid, thread_id).collection("messages")


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


def update_last_diagnosis(uid: str, thread_id: str,
                          cls: str, conf: float, image_ref: Optional[str] = None):
    """Thread'in son teşhis bilgisini güncelle"""
    tr = CLASS_TR.get(cls, cls)
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


def get_thread_data(uid: str, thread_id: str) -> dict:
    """Thread verilerini getir"""
    t_snap = thread_ref(uid, thread_id).get()
    return t_snap.to_dict() or {}
