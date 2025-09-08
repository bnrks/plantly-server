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

```
plantly-server/
├── app.py                          # FastAPI ana uygulama
├── start.txt                       # Sunucu başlatma komutları
├── models/
│   └── plant_disease_classifier_256.keras  # CNN modeli (ayrı indirin)
├── routers/                        # API endpoint'leri
│   ├── predict.py                  # Hastalık tespiti endpoint'i
│   ├── chat.py                     # HTTP chat endpoint'i
│   ├── ws_chat.py                  # WebSocket chat endpoint'i
│   └── server-secrets/             # Firebase kimlik bilgileri
│       └── plantly-admin.json
├── services/                       # İş mantığı servisleri
│   ├── predictService.py           # Model yükleme ve ön işleme
│   ├── auth/                       # Kimlik doğrulama
│   │   └── firebase_auth.py
│   ├── chat/                       # AI chat servisi
│   │   └── groq_service.py
│   ├── connection/                 # WebSocket yönetimi
│   │   └── websocket_manager.py
│   ├── database/                   # Veritabanı işlemleri
│   │   └── firestore_service.py
│   └── ml/                         # Makine öğrenmesi
│       └── prediction_service.py
└── ornek_yaprak.jpg                # Test görseli
```

### 🔧 Teknoloji Stack'i

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
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install fastapi uvicorn tensorflow pillow numpy python-dotenv
pip install firebase-admin google-cloud-firestore httpx pydantic
```

### 4. Model Dosyasını İndirin

🔗 **Model Dosyası**: [Google Drive Linki Buraya Eklenecek]

İndirilen `plant_disease_classifier_256.keras` dosyasını `models/` klasörüne yerleştirin.

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
  "answer": "{\"results\": {\"paragraph\": \"Bitkinizin Bakteriyel Lekelenme hastalığına yakalandığı tespit edildi.\", \"suggestions\": [\"Etkilenen yaprakları temizleyin\", \"Bakır içerikli fungisit uygulayın\"]}}"
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
  })
);

// Mesaj gönderme
ws.send(
  JSON.stringify({
    type: "text_message",
    content: "Bitkimin yaprağında lekeler var, ne yapmalıyım?",
  })
);

// Görsel gönderme (Base64)
ws.send(
  JSON.stringify({
    type: "image_message",
    image_data: "base64-encoded-image",
    content: "Bu yaprağa bir bakabilir misin?",
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
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

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

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun: `git checkout -b feature/yeni-ozellik`
3. Commit yapın: `git commit -m 'Yeni özellik eklendi'`
4. Push edin: `git push origin feature/yeni-ozellik`
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasını inceleyin.

## 👨‍💻 Geliştirici

**bnrks** - [GitHub Profile](https://github.com/bnrks)

## 🆘 Destek

Sorunlarınız için:

- **Issues**: [GitHub Issues](https://github.com/bnrks/plantly-server/issues)
- **Discussions**: Genel sorular ve tartışmalar
- **Email**: Kritik güvenlik sorunları için

## 📋 TODO

- [ ] Model performansı iyileştirmeleri
- [ ] Daha fazla hastalık desteği
- [ ] REST API v2
- [ ] Rate limiting
- [ ] Metrics ve monitoring
- [ ] Unit testler
- [ ] CI/CD pipeline

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
