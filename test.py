import requests
import json

# Ngrok URL
BASE_URL = "https://learning-partially-rabbit.ngrok-free.app"

def test_ping():
    """Ping endpoint'ini test et"""
    try:
        response = requests.get(f"{BASE_URL}/ping")
        print(f"Ping Status: {response.status_code}")
        print(f"Ping Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Ping Error: {e}")
        return False

def test_groq_chat():
    """Groq Chat endpoint'ini test et"""
    try:
        # Test prompt'u
        payload = {
            "prompt": "Domates bitkimin yaprakları sararıyor, ne yapmalıyım?"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print("Groq Chat endpoint'ine istek gönderiliyor...")
        response = requests.post(
            f"{BASE_URL}/groq-chat", 
            json=payload,
            headers=headers,
            timeout=60  # 60 saniye timeout
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Timeout Error: İstek 60 saniyede tamamlanmadı")
        return False
    except requests.exceptions.ConnectionError:
        print("Connection Error: Sunucuya bağlanılamadı")
        return False
    except Exception as e:
        print(f"Groq Chat Error: {e}")
        return False

def test_predict():
    """Predict endpoint'ini test et (varsa bir resim ile)"""
    try:
        # Workspace'te örnek resim var mı kontrol et
        import os
        image_files = ["ornek_yaprak.jpg", "ornek_yaprak2.jpg"]
        
        for img_file in image_files:
            if os.path.exists(img_file):
                print(f"{img_file} ile predict endpoint'ini test ediliyor...")
                
                with open(img_file, 'rb') as f:
                    files = {'file': (img_file, f, 'image/jpeg')}
                    response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
                
                print(f"Predict Status: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Predict Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
                else:
                    print(f"Predict Error: {response.text}")
                return response.status_code == 200
        
        print("Test için resim dosyası bulunamadı")
        return False
        
    except Exception as e:
        print(f"Predict Error: {e}")
        return False

if __name__ == "__main__":
    print("=== FastAPI Test Script ===")
    print(f"Testing URL: {BASE_URL}")
    print()
    
    # 1. Ping testi
    print("1. Ping Test:")
    ping_ok = test_ping()
    print(f"Ping Result: {'✓ OK' if ping_ok else '✗ FAIL'}")
    print()
    
    # 2. Groq Chat testi
    print("2. Groq Chat Test:")
    chat_ok = test_groq_chat()
    print(f"Groq Chat Result: {'✓ OK' if chat_ok else '✗ FAIL'}")
    print()
    
    # 3. Predict testi
    print("3. Predict Test:")
    predict_ok = test_predict()
    print(f"Predict Result: {'✓ OK' if predict_ok else '✗ FAIL'}")
    print()
    
    # Özet
    print("=== Test Summary ===")
    print(f"Ping: {'✓' if ping_ok else '✗'}")
    print(f"Groq Chat: {'✓' if chat_ok else '✗'}")
    print(f"Predict: {'✓' if predict_ok else '✗'}")