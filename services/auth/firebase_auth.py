# firebase_auth.py
# Firebase Authentication servisi

import os
from fastapi import HTTPException
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

load_dotenv()

# Firebase Configuration
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")

SA_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not SA_PATH:
    # routers/server-secrets klasörünü kontrol et
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services -> root
    candidate = os.path.join(here, "routers", "server-secrets", "plantly-admin.json")
    if os.path.exists(candidate):
        SA_PATH = candidate
        print(f"DEBUG: Firebase credentials found at: {SA_PATH}")
    else:
        print(f"DEBUG: Firebase credentials NOT found at: {candidate}")
        print(f"DEBUG: Files in directory: {os.listdir(os.path.dirname(candidate)) if os.path.exists(os.path.dirname(candidate)) else 'Directory does not exist'}")
else:
    print(f"DEBUG: Using environment variable GOOGLE_APPLICATION_CREDENTIALS: {SA_PATH}")

# Initialize Firebase Admin (singleton)
if not firebase_admin._apps:
    if SA_PATH and os.path.exists(SA_PATH):
        # Environment variable'ı set et ki diğer Google Cloud kütüphaneleri de kullansın
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SA_PATH
        cred = credentials.Certificate(SA_PATH)
        firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
    else:
        firebase_admin.initialize_app(options={'projectId': PROJECT_ID})


def verify_id_token_or_raise(id_token: str) -> str:
    """Firebase ID token'ı doğrula ve UID döndür"""
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


def get_project_id() -> str:
    """Firebase Project ID'yi döndür"""
    return PROJECT_ID
