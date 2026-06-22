# -*- coding: utf-8 -*-
"""
Ses Duygu Analizi (Hugging Face) - Video Destekli (Tek Model)
Yazar: ChatGPT (GPT-5)
Kullanım:
    python ses_duygu_analizi.py

Gereksinimler:
    pip install transformers torchaudio librosa huggingface_hub moviepy
"""

import os
import torch
import librosa
from huggingface_hub import login
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from moviepy import VideoFileClip

# ==================================================
# 1️⃣ Hugging Face Token'ını buraya yaz
# ==================================================
HF_TOKEN = "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"  #  kendi token'ını buraya yaz

# Token ile giriş yap
login(token=HF_TOKEN)

# ==================================================
# 2️⃣ Model Seçimi (manuel olarak değiştirebilirsin)
# ==================================================

# --- Mevcut Hugging Face SER Modelleri ---
# Türkçe:
# model_name = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"
# -- Güven skoru: 0.27 Sad
# model_name = "sefa-alper/wav2vec2-xlsr-turkish-speech-emotion-recognition-v3"
# --Tahmin edilen duygu: POSITIVE
# --Güven skoru: 0.34

# İngilizce:
# model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


# model_name = "r-f/wav2vec-english-speech-emotion-recognition"

# Genel / Çok-dilli / Diğer:
# model_name = "superb/wav2vec2-base-superb-er"
# Tahmin edilen duygu: hapy
#  Güven skoru: 0.75


# Çok dilli & ses anlayışı:
# model_name = "FunAudioLLM/SenseVoiceSmall"

#  Manuel olarak seçilen model
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# ==================================================
# 3️⃣ Yardımcı Fonksiyonlar
# ==================================================
def video_to_audio(video_path, audio_path="temp_audio.wav"):
    """MP4 videoyu al, WAV ses dosyasına çevir."""
    print(f"Video dosyası yükleniyor: {video_path}")
    video = VideoFileClip(video_path)
    # verbose ve logger parametreleri kaldırıldı
    video.audio.write_audiofile(audio_path, fps=16000)
    print(f"Ses çıkarıldı: {audio_path}")
    return audio_path


def load_audio(path, target_sr=16000):
    """Ses dosyasını yükle ve örnekleme oranını modele uygun hale getir."""
    audio, sr = librosa.load(path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def predict_emotion(model_name, audio_path, device=None):
    """Seçilen model ile sesin duygusunu tahmin et."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nModel yükleniyor: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForAudioClassification.from_pretrained(model_name, token=HF_TOKEN).to(device)

    audio = load_audio(audio_path)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Tahmin yapılıyor...")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label[predicted_id]
    score = torch.softmax(logits, dim=-1)[0, predicted_id].item()

    print("\n=== SONUÇ ===")
    print(f"Tahmin edilen duygu: {label}")
    print(f"Güven skoru: {score:.2f}")
    return label, score

# ==================================================
# 4️ Çalıştırma
# ==================================================
if __name__ == "__main__":
    print("=== Ses Duygu Analizi Başlatıldı ===")
    video_dosyasi = input("Analiz edilecek video dosyasının adını gir (örn: ornek.mp4): ").strip()

    if not os.path.exists(video_dosyasi):
        print(f"Dosya bulunamadı: {video_dosyasi}")
    else:
        # Video → WAV
        wav_path = video_to_audio(video_dosyasi)
        # Tek model ile tahmin
        predict_emotion(model_name, wav_path)
        # İsteğe bağlı: geçici WAV dosyasını silebilirsin
        # os.remove(wav_path)
