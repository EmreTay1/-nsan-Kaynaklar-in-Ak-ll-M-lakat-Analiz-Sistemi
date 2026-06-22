# Akıllı Mülakat Analiz ve Biyometrik Değerlendirme Sistemi

Bu proje, işe alım mülakat videolarını yapay zeka ile analiz ederek **biyometrik veriler** (ses tonu, yüz ifadesi), **konuşma metni** (transkripsiyon) ve **davranışsal işaretleri** (duraksamalar, dolgu kelimeleri) raporlayan kapsamlı bir karar destek sistemidir.

Sistem, kişisel verileri (PII) otomatik olarak anonimleştirir, detaylı analiz sonuçlarını raporlar ve işlem bitiminde hassas verileri **güvenli silme (secure wipe)** yöntemleriyle imha ederek KVKK uyumluluğunu sağlar.

## 🚀 Temel Özellikler

* **Çok Dilli Destek:** Türkçe ve İngilizce mülakatlar için dil seçeneği (Whisper ve dolgu kelime analizi buna göre çalışır).
* **Yüksek Doğruluklu Transkripsiyon:** OpenAI **Whisper Large-v3** modeli ile konuşmaların metne dökülmesi.
* **Duygu ve Biyometrik Analiz:**
* **Ses Tonu:** `wav2vec2` tabanlı modeller ile ses tonundan duygu (Öfkeli, Mutlu, Nötr vb.) tespiti.
* **Yüz İfadesi:** `DeepFace` kütüphanesi ile video üzerinden kare kare baskın duygu analizi.


* **Konuşmacı Ayrıştırma (Diarization):** `pyannote.audio` kullanarak Aday ve Mülakatçıyı birbirinden ayırma.
* **Davranışsal Metrikler:**
* Dolgu kelimeleri tespiti (ııı, hmm, şey / uh, um, like).
* Uzun duraksama (sessizlik) ve düşünme sürelerinin tespiti.


* **Veri Gizliliği (PII):** İsim, telefon, e-posta ve kurum bilgilerinin otomatik maskelenmesi (`[ADAY_ISMI]`, `[KURUM]` vb.).
* **Güvenlik ve Denetim:** Analiz sonrası ham dosyaların kalıcı imhası ve işlemin MSSQL veritabanına loglanması.

## 📂 Modül Yapısı ve İşlevleri

Proje modüler bir mimariye sahiptir. Her dosyanın görevi aşağıdadır:

### Ana Akış ve Arayüz

* **`gui.py`:** Kullanıcı arayüzüdür. Video seçimi, Dil seçimi (TR/EN) ve KVKK Açık Rıza onayı buradan yönetilir.
* **`biometric_pipeline.py`:** Sistemin beynidir. Tüm analiz süreçlerini (Ses çıkarma -> Diarization -> Transkript -> Raporlama -> İmha) sırasıyla yönetir.
* **`config.py`:** Tüm ayarların (Model isimleri, eşik değerleri, DB bağlantıları, API anahtarları) bulunduğu merkezi konfigürasyon dosyasıdır.

### İşleme Modülleri

* **`audio_processing.py`:** Videodan ses ayıklama (FFmpeg), konuşmacı ayrıştırma (Diarization), duraksama tespiti ve dolgu kelimesi (ııı, eee) analizi yapar.
* **`video_processing.py`:** Görüntü işleme modülüdür. Videoyu kare kare tarayarak yüz ifadelerini ve duyguları zaman damgasıyla çıkarır.
* **`transcription.py`:** Whisper modelini kullanarak sesi metne döker. Ses tonu, yüz ifadesi ve konuşmacı bilgisini metinle birleştirerek zenginleştirilmiş transkript oluşturur.
* **`audio_models.py`:** Ağır yapay zeka modellerini (Whisper, Emotion Recognition) belleğe yükler ve yönetir.

### Güvenlik ve Veri Yönetimi

* **`pii.py`:** Kişisel Verileri Koruma (PII) modülüdür. Metin içindeki özel isimleri ve hassas verileri yapay zeka ve Regex ile maskeler.
* **`io_ops.py`:** Dosya giriş/çıkış işlemlerini yönetir. En önemli görevi, işlem bitiminde geçici ve orijinal dosyaları **kurtarılamayacak şekilde** silmektir.
* **`db_log.py`:** Yapılan her işlemin (UUID, tarih, imha durumu) MSSQL veritabanındaki `ImhaLoglari` tablosuna kaydedilmesini sağlar.

---

## 🛠️ Kurulum

Bu projeyi çalıştırmak için aşağıdaki adımları izleyin.

### 1. Ön Gereksinimler

* **Python 3.10+**
* **FFmpeg:** Ses işleme için sistem yoluna (PATH) eklenmiş olmalıdır.
* **MSSQL Server:** (Opsiyonel) Loglama için yerel veya uzak sunucu.
* **GPU (Önerilen):** Modellerin hızlı çalışması için NVIDIA CUDA destekli ekran kartı.

### 2. Kütüphanelerin Yüklenmesi

Gerekli Python paketlerini yükleyin:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa pydub pyannote.audio deepface opencv-python pyodbc presidio-analyzer presidio-anonymizer tk

```

> **Not:** `pyannote.audio` kullanımı için Hugging Face hesabınızda kullanıcı sözleşmesini kabul etmeli ve bir Access Token almalısınız.

### 3. Yapılandırma (`config.py`)

`config.py` dosyasını açın ve gerekli alanları düzenleyin:

* `HF_TOKEN`: Hugging Face erişim jetonunuzu girin.
* `DB_SERVER`, `DB_USER`, `DB_PASSWORD`: MSSQL veritabanı bağlantı bilgilerinizi girin.

---

## ▶️ Kullanım

Uygulamayı başlatmak için terminalden arayüzü çalıştırın:

```bash
python gui.py

```

1. **Video Seç:** Analiz edilecek mülakat videosunu (.mp4, .avi) seçin.
2. **Dil Seç:** Mülakatın dilini (**Türkçe** veya **English**) seçin.
3. **Onay:** KVKK Açık Rıza metnini okuyup onay kutusunu işaretleyin.
4. **Başlat:** "Analizi Başlat" butonuna tıklayın.

İşlem tamamlandığında, analiz sonuçlarını içeren Word ve Metin dosyaları oluşturulacak, orijinal video ve geçici dosyalar otomatik olarak imha edilecektir.

---

## ⚠️ Yasal Uyarı ve Sorumluluk Reddi

Bu yazılım bir karar destek sistemidir. Üretilen analizler, duygusal tahminler ve metin dökümleri yapay zeka modelleri tarafından oluşturulmuştur ve %100 doğruluk garantisi vermez. Nihai işe alım kararı, insan kaynakları uzmanları tarafından verilmelidir. Sistem, KVKK ve GDPR uyumluluğu gözetilerek tasarlanmıştır ancak yasal yükümlülükler kullanıcının sorumluluğundadır.