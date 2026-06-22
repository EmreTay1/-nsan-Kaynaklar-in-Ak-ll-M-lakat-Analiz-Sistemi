#!/usr/bin/env python3
"""
Akıllı Mülakat Analizi + Ses Tonu (Duygu) Tespiti

- Konuşmacı ayırımı
- Whisper transkripsiyon (TR/EN)
- Duraksama, dolgu ve ses tonu etiketleme
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import whisper
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  #
import time

# ---------- AYARLAR ----------
VIDEO_DOSYASI = "WIN_20251013_13_29_35_Pro.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"
HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large"
# --- Mevcut Hugging Face SER Modelleri ---
# Türkçe:
# model_name = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"  kötü
# model_name = "sefa-alper/wav2vec2-xlsr-turkish-speech-emotion-recognition-v3" istenilen değil
#mpoyraz/wav2vec2-xls-r-300m-cv7-turkish -- çalısmıyor
# İngilizce:
# model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# model_name = "r-f/wav2vec-english-speech-emotion-recognition"


# Genel / Çok-dilli / Diğer:
# model_name = "superb/wav2vec2-base-superb-er, 4 duygu çalışması iyi eng ve tur
#superb/wav2vec2-large-superb-er


DUYGU_MODEL = "superb/wav2vec2-large-superb-er" #

# Çıktı dosyaları
JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

# Ses analizi parametreleri
DURAK_MIN_MS = 400
SES_ESIK_DB = -40
DOLGU_MIN_SN = 0.08
DOLGU_MAX_SN = 1.2
DOLGU_RMS_MIN = 0.001
DOLGU_RMS_MAX = 0.03

DOLGU_TR = ["(ııı)", "(eee)", "(hmm)", "(şey)"]
DOLGU_EN = ["(uh)", "(um)", "(hmm)", "(like)"]

# 🔹 Duygu modeli yükleme
print("Ses tonu modeli yükleniyor...")
extractor = AutoFeatureExtractor.from_pretrained(DUYGU_MODEL)
model_emotion = AutoModelForAudioClassification.from_pretrained(DUYGU_MODEL)
model_emotion.eval()


# ---------- FONKSİYONLAR ----------

def ses_tonu_tespit(wav_dosyasi):  # 🔹
    """Verilen WAV dosyasında ses tonunu (duygu) tahmin eder."""
    try:
        y, sr = librosa.load(wav_dosyasi, sr=16000)
        inputs = extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model_emotion(**inputs).logits
            tahmin = torch.argmax(logits, dim=1).item()
            etiket = model_emotion.config.id2label[tahmin]
        return etiket
    except Exception as e:
        return f"[Duygu tespit hatası: {e}]"


def ses_cikar(video_yolu, cikti_wav):
    print("\n[1/6] Videodan ses çıkarılıyor...")
    if Path(cikti_wav).exists():
        os.remove(cikti_wav)
        print("Eski ses dosyası silindi:", cikti_wav)
    cmd = ["ffmpeg", "-y", "-i", str(video_yolu), "-ac", "1", "-ar", "16000", "-vn", str(cikti_wav)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("Ses çıkarıldı:", cikti_wav)


def konusmaci_ayir(wav_yolu):
    print("\n[2/6] Konuşmacılar tespit ediliyor...")
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
    diarization = pipeline(str(wav_yolu))
    with open("output.rttm", "w") as f:
        diarization.write_rttm(f)
    print("Konuşmacılar belirlendi.")
    return diarization


def duraksama_tespit(wav_yolu):
    ses = AudioSegment.from_wav(wav_yolu)
    esik = ses.dBFS + SES_ESIK_DB
    sessiz_araliklar = silence.detect_silence(ses, min_silence_len=DURAK_MIN_MS, silence_thresh=esik)
    duraksamalar = [(s / 1000.0, e / 1000.0) for s, e in sessiz_araliklar]
    print("Duraksama sayisi bulundu:", len(duraksamalar))
    return duraksamalar


def dolgu_tespit(wav_yolu):
    print("Dolgu sesleri analiz ediliyor...")
    y, sr = librosa.load(wav_yolu, sr=16000)
    rms = librosa.feature.rms(y=y)[0]
    zaman = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    except Exception:
        f0 = np.full_like(rms, np.nan)
    maske = (~np.isnan(f0)) & (rms >= DOLGU_RMS_MIN) & (rms <= DOLGU_RMS_MAX)
    dolgu = []
    if maske.any():
        idx = np.where(maske)[0]
        gruplar = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in gruplar:
            bas, bit = zaman[g[0]], zaman[g[-1]]
            sure = bit - bas
            if DOLGU_MIN_SN <= sure <= DOLGU_MAX_SN:
                dolgu.append((round(bas, 3), round(bit, 3)))
    print("Dolgu sayisi bulundu:", len(dolgu))
    return dolgu


def transkripsiyon_yap(wav_yolu, diarization, dil):
    ses = AudioSegment.from_wav(str(wav_yolu))
    model = whisper.load_model(WHISPER_MODEL)
    transkript = []

    duraksamalar = duraksama_tespit(wav_yolu)
    dolgular = dolgu_tespit(wav_yolu)

    def segment_icindeki_dolgular(bas, bit):
        return [(s, e) for (s, e) in dolgular if s >= bas and s <= bit]

    def segment_oncesi_duraksama(bas):
        for (s, e) in duraksamalar:
            if e <= bas + 0.05 and e >= bas - 3.0:
                return (s, e)
        return None

    for tur, _, konusmaci in diarization.itertracks(yield_label=True):
        bas, bit = float(tur.start), float(tur.end)
        seg = ses[int(bas * 1000): int(bit * 1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            seg.export(tf.name, format="wav")
            gecici = tf.name

        print("Islenen segment:", konusmaci, f"[{bas:.2f}-{bit:.2f}]")

        try:
            sonuc = model.transcribe(gecici, language=dil)
            metin = sonuc.get("text", "").strip()
        except Exception as e:
            metin = f"[Transkripsiyon hatasi: {e}]"

        # 🔹 Ses tonunu tespit et
        ses_tonu = ses_tonu_tespit(gecici)

        os.remove(gecici)

        durak = segment_oncesi_duraksama(bas)
        if durak and (durak[1] - durak[0]) >= 0.7:
            metin = "(duraksama) " + metin

        seg_dolgu = segment_icindeki_dolgular(bas, bit)
        if seg_dolgu and metin.strip():
            kelimeler = metin.split()
            uzunluk = max(0.0001, bit - bas)
            seg_dolgu.sort(key=lambda x: x[0])
            etiketler = DOLGU_TR if dil == "tr" else DOLGU_EN
            for (ds, de) in seg_dolgu[:2]:  # 🔹 max 2 dolgu
                rel = (ds - bas) / uzunluk
                pos = int(rel * len(kelimeler))
                pos = max(0, min(len(kelimeler), pos))
                kelimeler.insert(pos, np.random.choice(etiketler))
            metin = " ".join(kelimeler)

        transkript.append({
            "konusmaci": konusmaci,
            "bas": bas,
            "bit": bit,
            "metin": metin,
            "ses_tonu": ses_tonu,  # 🔹 yeni alan
            "duraksama": round((durak[1] - durak[0]) if durak else 0.0, 3),
            "dolgu_sayisi": len(seg_dolgu)
        })

    print("Transkripsiyon + ses tonu analizi tamamlandi.")
    return transkript


def json_kaydet(veri, dosya=JSON_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)
    print("JSON kaydedildi:", dosya)


def txt_kaydet(veri, dosya=TXT_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        for s in veri:
            f.write(f"[{s['konusmaci']} | {s['ses_tonu']}]: {s['metin']}\n\n")  # 🔹 ses tonu yaz
    print("TXT kaydedildi:", dosya)


def istatistik_goster(transkript):
    print("\nMülakat Ozeti")
    toplam_dolgu = sum(s["dolgu_sayisi"] for s in transkript)
    toplam_durak = sum(s["duraksama"] for s in transkript)
    ses_tonlari = {}
    for s in transkript:
        ton = s["ses_tonu"]
        ses_tonlari[ton] = ses_tonlari.get(ton, 0) + 1
    print(f"Toplam duraksama suresi: {toplam_durak:.2f} sn")
    print(f"Toplam dolgu sayisi: {toplam_dolgu}")
    print("Ses tonu dağılımı:", ses_tonlari)


def calistir(dil="tr"):
    print("Akilli Mulakat Analizi Baslatiliyor...")
    ses_cikar(VIDEO_DOSYASI, WAV_DOSYASI)
    diarization = konusmaci_ayir(WAV_DOSYASI)
    transkript = transkripsiyon_yap(WAV_DOSYASI, diarization, dil)
    json_kaydet(transkript)
    txt_kaydet(transkript)
    istatistik_goster(transkript)
    print("Analiz tamamlandi.")


def main():
    print("\nMulakat dili secin:")
    print("1) Turkce")
    print("2) İngilizce")
    secim = input("Secim (1/2): ").strip()
    dil = "tr" if secim == "1" else "en"
    calistir(dil)


if __name__ == "__main__":
    main()
