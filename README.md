# 🌱 Plantly Server

**Plantly Server**, bitki hastalık tespiti ve yapay zeka destekli bitki bakım danışmanlığı sunan FastAPI tabanlı bir backend servisidir. Derin öğrenme modeli ile bitki hastalıklarını tespit eder ve kullanıcılara Groq AI ile kişiselleştirilmiş bakım önerileri sunar.

## 🚀 Özellikler

### 🔬 Bitki Hastalık Tespiti

- **CNN Modeli**: 256x256 çözünürlükte görüntü analizi
- **Desteklenen Hastalıklar**:
  - Bakteriyel Lekelenme (Bacterial Spot)
  - Erken Yanıklık (Early Blight)
  - Geç Yanıklık (Late Blight)
  - Sağlıklı Bitki Tespiti
- **Güven Skoru**: Her tahmin için detaylı güvenilirlik oranı
- **Hızlı İşlem**: Milisaniye düzeyinde tahmin süresi

### 🤖 AI Destekli Chat Sistemi

- **Groq AI Entegrasyonu**: Gelişmiş dil modeli ile bitki bakım danışmanlığı
- **Türkçe Destek**: Hastalık açıklamaları ve öneriler Türkçe
- **Kişiselleştirilmiş Öneriler**: Tespit edilen hastalığa özel bakım rehberi
- **Real-time Chat**: WebSocket destekli anlık sohbet

### 💾 Veri Yönetimi

- **Firebase Firestore**: Kullanıcı verileri ve sohbet geçmişi
- **Firebase Auth**: Güvenli kullanıcı kimlik doğrulama
- **Thread Sistemi**: Organize sohbet geçmişi
- **Hafıza Yönetimi**: Bağlamsal sohbet sürekliliği

### 🔄 WebSocket Desteği

- **Gerçek Zamanlı İletişim**: Anlık mesajlaşma
- **Çoklu Kullanıcı**: Eşzamanlı bağlantı yönetimi
- **Oturum Yönetimi**: Güvenli bağlantı kontrolü

## 🏗️ Sistem Mimarisi

### 📁 Proje Yapısı

````
plantly-server/
├── .env                            # Ortam değişkenleri
├── .gitignore                      # Git ignore dosyası
├── app.py                          # FastAPI ana uygulama
├── requirements.txt                # Python bağımlılıkları
├── start.txt                       # Sunucu başlatma komutları
├── test.py                         # Test dosyası
├── ornek_yaprak.jpg                # Test görseli
├── ml/
│   ├── classes/
│   │   └── classes.json                   # Model class listesi
│   └── models/
│       └── mobilenetv2_final.keras        # CNN inference modeli
├── routers/                        # API endpoint'leri
│   ├── predict.py                  # Hastalık tespiti endpoint'i
│   ├── chat.py                     # HTTP chat endpoint'i
│   ├── ws_chat.py                  # WebSocket chat endpoint'i
│   └── server-secrets/             # Firebase kimlik bilgileri
│       └── plantly-admin.json
└── services/                       # İş mantığı servisleri
    ├── __init__.py
    ├── predictService.py           # Model yükleme ve ön işleme
    ├── auth/                       # Kimlik doğrulama
    │   ├── __init__.py
    │   └── firebase_auth.py
    ├── chat/                       # AI chat servisi
    │   ├── __init__.py
    │   └── groq_service.py
    ├── connection/                 # WebSocket yönetimi
    │   ├── __init__.py
    │   └── websocket_manager.py
    ├── database/                   # Veritabanı işlemleri
    │   ├── __init__.py
    │   └── firestore_service.py
    └── ml/                         # Makine öğrenmesi
        ├── __init__.py
        └── prediction_service.py
```### 🔧 Teknoloji Stack'i

#### Backend Framework

- **FastAPI**: Modern, hızlı Python web framework
- **Uvicorn**: ASGI server
- **WebSocket**: Real-time communication

#### Machine Learning

- **TensorFlow/Keras**: Derin öğrenme framework'ü
- **PIL (Pillow)**: Görüntü işleme
- **NumPy**: Sayısal hesaplamalar

#### AI & Chat

- **Groq AI**: Gelişmiş dil modeli
- **HTTPX**: Async HTTP client

#### Database & Auth

- **Firebase Firestore**: NoSQL veritabanı
- **Firebase Authentication**: Kullanıcı yönetimi
- **Google Cloud**: Cloud servisleri

#### Utilities

- **Python-dotenv**: Ortam değişkenleri yönetimi
- **Pydantic**: Veri validasyonu

## 🛠️ Kurulum

### Ön Gereksinimler

- Python 3.8+
- pip
- Firebase projesi
- Groq AI API anahtarı

### 1. Repository'yi Klonlayın

```bash
git clone https://github.com/bnrks/plantly-server.git
cd plantly-server
````

### 2. Sanal Ortam Oluşturun

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

> Not: `ml/models/mobilenetv2_final.keras` için TensorFlow/Keras sürüm uyumluluğu gerekir.
> `requirements.txt` içindeki TensorFlow sürümünü kullanın.

Ya da manuel olarak:

```bash
pip install fastapi uvicorn tensorflow pillow numpy python-dotenv
pip install firebase-admin google-cloud-firestore httpx pydantic
pip install python-multipart websockets
```

### 4. Model Dosyasını Hazırlayın

- Inference modeli: `ml/models/mobilenetv2_final.keras`
- Class listesi: `ml/classes/classes.json`

### 5. Firebase Konfigürasyonu

1. Firebase projenizi oluşturun
2. Service Account anahtarını indirin
3. `routers/server-secrets/plantly-admin.json` olarak kaydedin

### 6. Ortam Değişkenlerini Ayarlayın

`.env` dosyası oluşturun:

```env
# Firebase
FIREBASE_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=routers/server-secrets/plantly-admin.json

# Groq AI
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b

# Chat Memory Settings
HISTORY_MAX_CHARS=8000
MEMORY_ENABLED=1
MEMORY_REFRESH_EVERY=3
MEM_FACTS_LIMIT=8
```

### 7. Sunucuyu Başlatın

```bash
# Geliştirme modu
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Üretim modu
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📚 API Kullanımı

### 🔍 Hastalık Tespiti

```http
POST /predict
Content-Type: multipart/form-data

file: [bitki_gorseli.jpg]
```

**Yanıt:**

```json
{
  "class": "bacterial_spot",
  "confidence": 0.87,
  "probs": [0.87, 0.08, 0.03, 0.02],
  "latency_ms": 245
}
```

### 💬 Chat API

```http
POST /groq-chat
Content-Type: application/json

{
    "prompt": "class: bacterial_spot, confidence: 87%"
}
```

**Yanıt:**

```json
{
  "answer": "{\"content\": \"Bitkinizin Bakteriyel Lekelenme hastalığına yakalandığı tespit edildi. Bu hastalık yapraklarda kahverengi lekeler oluşturur ve zamanında müdahale edilmezse bitkiyi ciddi şekilde etkileyebilir.\", \"notes\": [\"Etkilenen yaprakları temizleyin ve imha edin\", \"Bakır içerikli fungisit uygulayın\", \"Sulamayı yapraklara değmeyecek şekilde topraktan yapın\", \"Bitki çevresindeki hava sirkülasyonunu artırın\"]}"
}
```

### 🔄 WebSocket Chat

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/chat");

// Bağlantı kurma
ws.send(
  JSON.stringify({
    type: "init",
    idToken: "firebase-id-token",
    thread_id: "optional-thread-id",
    new_thread: false, // yeni thread oluşturmak için true
  })
);

// Metin mesajı gönderme
ws.send(
  JSON.stringify({
    type: "user_text",
    text: "Bitkimin yaprağında lekeler var, ne yapmalıyım?",
  })
);

// Teşhis mesajı (görsel analiz sonucu)
ws.send(
  JSON.stringify({
    type: "diagnosis",
    class: "bacterial_spot",
    confidence: 0.87,
    image_ref: "optional-image-reference",
    auto_reply: true,
  })
);

// Ping mesajı
ws.send(
  JSON.stringify({
    type: "ping",
  })
);
```

## 🎯 Endpoint'ler

| Method | Endpoint     | Açıklama                   |
| ------ | ------------ | -------------------------- |
| GET    | `/`          | Sunucu durumu              |
| GET    | `/ping`      | Health check               |
| POST   | `/predict`   | Bitki hastalığı tespiti    |
| POST   | `/groq-chat` | AI chat (HTTP)             |
| WS     | `/ws/chat`   | Real-time chat (WebSocket) |

## 🔒 Güvenlik

- **Firebase Auth**: Tüm kullanıcı işlemleri kimlik doğrulaması gerektirir
- **Input Validation**: Pydantic ile veri doğrulama
- **Error Handling**: Kapsamlı hata yönetimi
- **CORS**: Güvenli cross-origin istekler

## 🚀 Deployment

### Docker ile (Önerilen)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Port'u aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms

- **Google Cloud Run**: Serverless deployment
- **AWS EC2**: Geleneksel sunucu
- **Heroku**: Hızlı deployment
- **Railway**: Modern cloud platform

## 🧪 Test

```bash
# Test dosyası ile prediction testi
curl -X POST "http://localhost:8000/predict" \
     -F "file=@ornek_yaprak.jpg"

# Health check
curl http://localhost:8000/ping
```

## 📈 Performance

- **Model İnference**: ~250ms
- **API Response**: <500ms
- **WebSocket Latency**: <100ms
- **Memory Usage**: ~2GB (model dahil)

## 🔧 Geliştirme

### Code Style

- **PEP 8**: Python style guide
- **Type Hints**: Tür bilgisi ekleme
- **Docstrings**: Fonksiyon dokümantasyonu

### Debugging

```bash
# Debug modu ile başlatma
python -c "import app; print('Debug mode')"
uvicorn app:app --reload --log-level debug
```

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasını inceleyin.

## 👨‍💻 Geliştirici

**bnrks** - [GitHub Profile](https://github.com/bnrks)

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
