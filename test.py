#!/usr/bin/env python3
"""
Plantly Server Test Script - Thread Chat System
Usage: python test.py
"""
import requests
import json
import time
import os

BASE_URL = "https://learning-partially-rabbit.ngrok-free.app"

def test_ping():
    """Ping endpoint testi"""
    try:
        response = requests.get(f"{BASE_URL}/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_thread_system():
    """Thread sistemi kapsamlı testi"""
    try:
        print("\n🧵 Thread Sistemi Testi")
        
        # 1. Yeni thread oluştur
        print("1. Thread oluşturuluyor...")
        create_response = requests.post(
            f"{BASE_URL}/threads",
            json={"title": "Test Bitki Bakımı Sohbeti"}
        )
        if create_response.status_code != 200:
            print(f"❌ Thread oluşturma hatası: {create_response.text}")
            return False
        
        thread_data = create_response.json()
        thread_id = thread_data["thread_id"]
        print(f"✅ Thread oluşturuldu: {thread_id}")
        
        # 2. Thread'e mesaj gönder
        print("2. Mesaj gönderiliyor...")
        message_response = requests.post(
            f"{BASE_URL}/threads/{thread_id}/messages",
            json={
                "thread_id": thread_id,
                "message": "Merhaba! Domates bitkimin yaprakları sararıyor, ne yapmalıyım?"
            }
        )
        if message_response.status_code != 200:
            print(f"❌ Mesaj gönderme hatası: {message_response.text}")
            return False
        
        message_data = message_response.json()
        print(f"✅ AI Yanıt: {message_data['ai_response']['content'][:100]}...")
        
        # 3. Thread detayını getir
        print("3. Thread detayı getiriliyor...")
        detail_response = requests.get(f"{BASE_URL}/threads/{thread_id}")
        if detail_response.status_code != 200:
            print(f"❌ Thread detay hatası: {detail_response.text}")
            return False
        
        detail_data = detail_response.json()
        print(f"✅ Thread: {len(detail_data['messages'])} mesaj içeriyor")
        
        # 4. Tüm thread'leri listele
        print("4. Thread listesi getiriliyor...")
        list_response = requests.get(f"{BASE_URL}/threads")
        if list_response.status_code != 200:
            print(f"❌ Thread listesi hatası: {list_response.text}")
            return False
        
        list_data = list_response.json()
        print(f"✅ Toplam {list_data['total']} thread bulundu")
        
        # 5. Thread'i sil
        print("5. Thread siliniyor...")
        delete_response = requests.delete(f"{BASE_URL}/threads/{thread_id}")
        if delete_response.status_code != 200:
            print(f"❌ Thread silme hatası: {delete_response.text}")
            return False
        
        print("✅ Thread başarıyla silindi")
        return True
        
    except Exception as e:
        print(f"❌ Thread sistemi hatası: {e}")
        return False

def test_predict():
    """Predict endpoint testi"""
    try:
        for img_file in ["ornek_yaprak.jpg", "ornek_yaprak2.jpg"]:
            if os.path.exists(img_file):
                with open(img_file, 'rb') as f:
                    files = {'file': (img_file, f, 'image/jpeg')}
                    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Predict: {result['class']} ({result['confidence']:.2%})")
                        return True
                    else:
                        print(f"❌ Predict error: {response.status_code}")
                        return False
        print("❌ Test resmi bulunamadı")
        return False
    except Exception as e:
        print(f"❌ Predict error: {e}")
        return False

def test_groq_chat():
    """Eski Groq chat endpoint testi"""
    try:
        payload = {"prompt": "Çiçeklerim neden soluk renkte?"}
        response = requests.post(f"{BASE_URL}/groq-chat", json=payload, timeout=30)
        if response.status_code == 200:
            print("✅ Groq chat çalışıyor")
            return True
        else:
            print(f"❌ Groq chat error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Groq chat error: {e}")
        return False

if __name__ == "__main__":
    print("🌱 Plantly Server Test Suite")
    print(f"Testing: {BASE_URL}")
    print("=" * 50)
    
    tests = [
        ("Ping", test_ping),
        ("Predict", test_predict), 
        ("Groq Chat (Legacy)", test_groq_chat),
        ("Thread System", test_thread_system)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        result = test_func()
        results.append(result)
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}")
    
    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 Tüm testler başarılı!")
    else:
        print("⚠️  Bazı testler başarısız oldu.")