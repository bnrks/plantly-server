# websocket_manager.py
# WebSocket bağlantı yöneticisi

import json
from typing import Dict, Set, Any
from fastapi import WebSocket


class WebSocketManager:
    """WebSocket bağlantılarını yönetir"""
    
    def __init__(self):
        self.rooms: Dict[str, Set[WebSocket]] = {}

    async def connect(self, thread_id: str, websocket: WebSocket):
        """WebSocket'i belirli bir thread'e bağla"""
        # accept() sadece chat_ws içinde yapılır
        self.rooms.setdefault(thread_id, set()).add(websocket)

    def disconnect(self, thread_id: str, websocket: WebSocket):
        """WebSocket bağlantısını kapat"""
        if thread_id in self.rooms and websocket in self.rooms[thread_id]:
            self.rooms[thread_id].remove(websocket)
        if thread_id in self.rooms and not self.rooms[thread_id]:
            self.rooms.pop(thread_id, None)

    async def broadcast(self, thread_id: str, payload: Dict[str, Any]):
        """Thread'deki tüm WebSocket'lere mesaj gönder"""
        dead = []
        for ws in self.rooms.get(thread_id, []):
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        
        # Ölü bağlantıları temizle
        for ws in dead:
            self.disconnect(thread_id, ws)

    def get_active_connections(self, thread_id: str) -> int:
        """Thread'deki aktif bağlantı sayısını döndür"""
        return len(self.rooms.get(thread_id, set()))

    def get_all_active_threads(self) -> list:
        """Aktif thread'lerin listesini döndür"""
        return list(self.rooms.keys())
