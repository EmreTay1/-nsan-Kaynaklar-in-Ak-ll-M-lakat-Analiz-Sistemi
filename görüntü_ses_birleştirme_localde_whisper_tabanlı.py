#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Akıllı Mülakat Analizi + Ses Tonu ve Yüz Duygusu Tespiti

ÖZET:
- Videodan ses çıkarılır (16 kHz mono WAV)
- Hugging Face Whisper pipeline ile transkripsiyon yapılır
- Pyannote ile konuşmacı ayrımı yapılır
- Duraksama ve dolgu tespiti yapılır
- DeepFace ile video karelerinden dominant yüz ifadeleri tespit edilir
- Tüm zaman damgaları segmentlere eşleştirilir
- JSON ve TXT çıktısı üretilir
- TXT çıktısında format: [Konuşmacı | Ses tonu: … | Yüz ifadesi: …] Metin
"""

# ----------------------------- KÜTÜPHANELER -----------------------------
import os
import subprocess
import tempfile
import json
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
from pydub import AudioSegment, silence
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from pyannote.audio import Pipeline
import cv2
from deepface import DeepFace
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ----------------------------- AYARLAR -----------------------------
VIDEO_DOSYASI = "video1723838072.mp4"    # İşlenecek video dosyası
WAV_DOSYASI = "gecici_16k_mono.wav"     # Geçici ses dosyası

HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

WHISPER_MODEL_ID = "openai/whisper-large-v3"
DUYGU_MODEL = "superb/wav2vec2-large-superb-er"

JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

SES_ESIK_DB = -40
DURAK_MIN_MS = 400

DOLGU_MIN_SN = 0.12
DOLGU_MAX_SN = 0.60
DOLGU_RMS_MIN = 0.005
DOLGU_RMS_MAX = 0.020
DOLGU_MERGE_GAP = 0.25
MAX_DOLGU_PER_SEGMENT = 1
DOLGU_TR = ["(ııı)", "(eee)", "(hmm)", "(şey)"]
DOLGU_EN = ["(uh)", "(um)", "(hmm)", "(like)"]

DUYGU_SOZLUGU = {
    'angry': 'ÖFKELİ',
    'disgust': 'TİKSİNMİŞ',
    'fear': 'KORKMUŞ',
    'happy': 'MUTLU',
    'sad': 'ÜZGÜN',
    'surprise': 'ŞAŞKIN',
    'neutral': 'NÖTR'
}

# ----------------------------- MODELLER -----------------------------
print("Ses tonu modeli yükleniyor...")
try:
    extractor = AutoFeatureExtractor.from_pretrained(DUYGU_MODEL)
    model_emotion = AutoModelForAudioClassification.from_pretrained(DUYGU_MODEL)
    model_emotion.eval()
except Exception as e:
    print(f"Duygu modeli yüklemede hata: {e}")
    extractor = None
    model_emotion = None

print(f"Whisper modeli yükleniyor: {WHISPER_MODEL_ID}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model_whisper.to(device)
    processor_whisper = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_whisper,
        tokenizer=processor_whisper.tokenizer,
        feature_extractor=processor_whisper.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
except Exception as e:
    print(f"Whisper modeli yüklenemedi: {e}")
    transcriber = None

# ----------------------------- FONKSİYONLAR -----------------------------

def ses_cikar(video_yolu, cikti_wav):
    """Videodan ses çıkarır (16 kHz mono WAV)."""
    print("[1/7] Videodan ses çıkarılıyor...")
    if Path(cikti_wav).exists():
        os.remove(cikti_wav)
    cmd = ["ffmpeg", "-y", "-i", str(video_yolu), "-ac", "1", "-ar", "16000", "-vn", str(cikti_wav)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Ses çıkarıldı: {cikti_wav}")

def konusmaci_ayir(wav_yolu):
    """Pyannote ile konuşmacıları tespit eder."""
    print("[2/7] Konuşmacılar tespit ediliyor...")
    pipeline_diarize = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
    diarization = pipeline_diarize(str(wav_yolu))
    return diarization

def duraksama_tespit(wav_yolu):
    """Duraksama / sessizlik tespiti."""
    ses = AudioSegment.from_wav(wav_yolu)
    esik = ses.dBFS + SES_ESIK_DB
    sessiz_araliklar = silence.detect_silence(ses, min_silence_len=DURAK_MIN_MS, silence_thresh=esik)
    return [(s/1000.0, e/1000.0) for s,e in sessiz_araliklar]

def dolgu_tespit(wav_yolu):
    """Dolgu seslerini tespit eder (optimize)."""
    y, sr = librosa.load(wav_yolu, sr=16000)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zaman = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_time = np.linspace(0, len(y)/sr, num=len(f0))
        f0_interp = np.interp(zaman, f0_time, np.nan_to_num(f0, nan=0.0))
        voiced_frame_mask = f0_interp>0.0
    except Exception:
        voiced_frame_mask = np.zeros_like(rms, dtype=bool)
    rms_mask = (rms>=DOLGU_RMS_MIN) & (rms<=DOLGU_RMS_MAX)
    maske = rms_mask & voiced_frame_mask
    dolgu=[]
    if maske.any():
        idx = np.where(maske)[0]
        gruplar = np.split(idx, np.where(np.diff(idx)!=1)[0]+1)
        for g in gruplar:
            bas, bit = zaman[g[0]], zaman[g[-1]]
            sure = bit-bas
            if DOLGU_MIN_SN<=sure<=DOLGU_MAX_SN:
                dolgu.append((round(bas,3), round(bit,3)))
    return dolgu

# ----------------------------- YÜZ İFADESİ -----------------------------
def yuz_ifadesi_analiz(video_yolu, saniye_araligi=0.5):
    """Video üzerinden yüz ifadelerini tespit eder."""
    print("[3/7] Yüz ifadeleri analizi başlatılıyor...")
    video = cv2.VideoCapture(video_yolu)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * saniye_araligi)
    yuz_cizelgesi = []

    frame_num = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(analysis, list) and len(analysis)>0:
                    dominant_en = analysis[0]['dominant_emotion']
                    dominant_tr = DUYGU_SOZLUGU.get(dominant_en, dominant_en.upper())
                    timestamp = frame_num / fps
                    yuz_cizelgesi.append({'zaman': timestamp, 'duygu': dominant_tr})
            except Exception:
                pass
        frame_num += 1
    video.release()
    print("Yüz ifadeleri analizi tamamlandı.")
    return yuz_cizelgesi

# ----------------------------- TRANSKRİPSİYON -----------------------------
def transkripsiyon_yap(wav_yolu, diarization, dil, yuz_cizelgesi):
    """Transkripsiyon yapar, dolgu, duraksama, ses tonu ve yüz ifadelerini ekler."""
    ses = AudioSegment.from_wav(str(wav_yolu))
    transkript=[]
    duraksamalar = duraksama_tespit(wav_yolu)
    dolgular = dolgu_tespit(wav_yolu)

    def segment_icindeki_dolgular(bas, bit):
        return [(s,e) for (s,e) in dolgular if s>=bas and s<=bit]

    def segment_yuz(bas, bit):
        return [y['duygu'] for y in yuz_cizelgesi if bas <= y['zaman'] <= bit]

    def segment_oncesi_duraksama(bas):
        for s,e in duraksamalar:
            if e<=bas+0.05 and e>=bas-3.0:
                return (s,e)
        return None

    for tur, _, konusmaci in diarization.itertracks(yield_label=True):
        bas, bit = float(tur.start), float(tur.end)
        seg = ses[int(bas*1000): int(bit*1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            seg.export(tf.name, format="wav")
            gecici=tf.name
        # --- Transkripsiyon ---
        try:
            if transcriber is None:
                raise Exception("Whisper modeli yüklenemedi.")
            sonuc = transcriber(gecici, generate_kwargs={"language":dil})
            metin = sonuc.get("text","").strip()
        except Exception as e:
            metin=f"[Transkripsiyon hatasi: {e}]"

        # --- Duraksama ekle ---
        durak = segment_oncesi_duraksama(bas)
        if durak and (durak[1]-durak[0])>=0.7:
            metin="(duraksama) "+metin

        # --- Ses tonu ---
        ses_tonu="[N/A]"
        if extractor and model_emotion:
            try:
                ytmp, srtmp = librosa.load(gecici, sr=16000)
                inputs = extractor(ytmp, sampling_rate=srtmp, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model_emotion(**inputs).logits
                    tahmin = torch.argmax(logits, dim=1).item()
                    ses_tonu = model_emotion.config.id2label[tahmin]
            except Exception:
                ses_tonu="[HATA]"

        # --- Yüz ifadesi ---
        yuzler = segment_yuz(bas, bit)
        yuz_ifadesi = Counter(yuzler).most_common(1)[0][0] if yuzler else "BELİRSİZ"

        # --- Dolgu ekle ---
        seg_dolgu = segment_icindeki_dolgular(bas, bit)
        kelime_sayisi = len(metin.split())
        if seg_dolgu and metin.strip() and kelime_sayisi>=2:
            seg_dolgu_sorted = sorted(seg_dolgu, key=lambda x:x[1]-x[0], reverse=True)
            chosen = seg_dolgu_sorted[:MAX_DOLGU_PER_SEGMENT]
            kelimeler = metin.split()
            uzunluk = max(0.0001, bit-bas)
            insert_positions=[]
            for ds,de in chosen:
                rel_pos = (ds-bas)/uzunluk
                word_pos=int(rel_pos*len(kelimeler))
                insert_positions.append(min(len(kelimeler), word_pos))
            for i,pos in enumerate(sorted(insert_positions)):
                etiket = np.random.choice(DOLGU_TR if dil=="tr" else DOLGU_EN)
                kelimeler.insert(pos+i, etiket)
            metin=" ".join(kelimeler)

        os.remove(gecici)

        transkript.append({
            "konusmaci": konusmaci,
            "bas": round(bas,3),
            "bit": round(bit,3),
            "metin": metin,
            "ses_tonu": ses_tonu,
            "yuz_ifadesi": yuz_ifadesi
        })
    print("[4/7] Transkripsiyon + analiz tamamlandı.")
    return transkript

# ----------------------------- ÇIKTI -----------------------------
def json_kaydet(veri, dosya=JSON_CIKTI):
    with open(dosya,"w",encoding="utf-8") as f:
        json.dump(veri,f,ensure_ascii=False, indent=2)
    print(f"[5/7] JSON kaydedildi: {dosya}")

def txt_kaydet(veri, dosya=TXT_CIKTI):
    with open(dosya,"w",encoding="utf-8") as f:
        for s in veri:
            f.write(f"[{s['konusmaci']} | Ses tonu: {s['ses_tonu']} | Yüz ifadesi: {s['yuz_ifadesi']}]: {s['metin']}\n\n")
    print(f"[6/7] TXT kaydedildi: {dosya}")

# ----------------------------- ÇALIŞTIR -----------------------------
def calistir(dil="tr"):
    print("Akıllı Mülakat Analizi Başlatılıyor...")
    ses_cikar(VIDEO_DOSYASI, WAV_DOSYASI)
    diarization = konusmaci_ayir(WAV_DOSYASI)
    yuz_cizelgesi = yuz_ifadesi_analiz(VIDEO_DOSYASI)
    transkript = transkripsiyon_yap(WAV_DOSYASI, diarization, dil, yuz_cizelgesi)
    json_kaydet(transkript)
    txt_kaydet(transkript)
    print("[7/7] Analiz tamamlandı.")

def main():
    print("Mülakat dilini seçin: 1) Türkçe  2) İngilizce")
    secim = input("Seçim (1/2): ").strip()
    dil = "tr" if secim=="1" else "en"
    calistir(dil)

if __name__=="__main__":
    main()
