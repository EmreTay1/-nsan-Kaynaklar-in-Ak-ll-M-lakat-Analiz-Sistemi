#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akıllı Mülakat Analizi + Ses Tonu (Duygu) + Yüz İfadesi Analizi
Yapı aynı kalır, sadece görüntü analizi (DeepFace) eklendi.
Zaman damgalı yüz ifadesi çıkarımı yapılır.
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from deepface import DeepFace
import cv2

# ---------- AYARLAR ----------
VIDEO_DOSYASI = "video1723838072.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"

HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL_ID = "Huseyin/whisper-large-v3-turkish-finetuned"
DUYGU_MODEL = "superb/wav2vec2-large-superb-er"

JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

# ----------------- Model yüklemeleri -----------------
print("Duygu modeli yükleniyor...")
extractor = AutoFeatureExtractor.from_pretrained(DUYGU_MODEL)
model_emotion = AutoModelForAudioClassification.from_pretrained(DUYGU_MODEL)
model_emotion.eval()

print(f"Whisper modeli yükleniyor: {WHISPER_MODEL_ID} ...")
use_cuda = torch.cuda.is_available()
device_id = 0 if use_cuda else -1
torch_dtype = torch.float16 if use_cuda else torch.float32

model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
processor_whisper = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
if use_cuda:
    model_whisper.to(f"cuda:{device_id}")

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_whisper,
    tokenizer=processor_whisper.tokenizer,
    feature_extractor=processor_whisper.feature_extractor,
    chunk_length_s=30,
    batch_size=8,
    device=device_id,
)
print("Whisper pipeline başarıyla yüklendi.")


# ----------------- GÖRÜNTÜ ANALİZİ -----------------
def yuz_ifadesi_analiz(video_yolu, fps_orani=1):
    """
    Videodan her 1 saniyede 1 kare alır, DeepFace ile yüz ifadesi analiz eder.
    Her analizin zaman damgasını (saniye cinsinden) döndürür.
    """
    print("\n[0/6] Görüntü analizi (yüz ifadesi) başlatılıyor...")
    cap = cv2.VideoCapture(video_yolu)
    fps = cap.get(cv2.CAP_PROP_FPS)
    toplam_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    toplam_sure = toplam_frame / fps

    frame_interval = int(fps * fps_orani)
    zaman_cizelgesi = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            saniye = frame_id / fps
            try:
                sonuc = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(sonuc, list):
                    sonuc = sonuc[0]
                ifade = sonuc.get("dominant_emotion", "NÖTR").upper()
            except Exception:
                ifade = "NÖTR"
            zaman_cizelgesi.append({"time": round(saniye, 2), "yuz_ifadesi": ifade})
        frame_id += 1
    cap.release()
    print(f"Yüz ifadesi analizi tamamlandı ({len(zaman_cizelgesi)} örnek).")
    return zaman_cizelgesi


def en_yakin_yuz_ifadesi(yuz_cizelgesi, bas, bit):
    """
    Segment zaman aralığına en yakın yüz ifadesini döndürür.
    """
    hedef = (bas + bit) / 2
    farklar = [abs(x["time"] - hedef) for x in yuz_cizelgesi]
    if not farklar:
        return "NÖTR"
    return yuz_cizelgesi[int(np.argmin(farklar))]["yuz_ifadesi"]


# ----------------- SES İŞLEME -----------------
def ses_cikar(video_yolu, cikti_wav):
    print("\n[1/6] Videodan ses çıkarılıyor...")
    if Path(cikti_wav).exists():
        os.remove(cikti_wav)
    cmd = ["ffmpeg", "-y", "-i", str(video_yolu), "-ac", "1", "-ar", "16000", "-vn", str(cikti_wav)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Ses çıkarıldı: {cikti_wav}")


def konusmaci_ayir(wav_yolu):
    print("\n[2/6] Konuşmacılar tespit ediliyor...")
    pipeline_diarize = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
    diarization = pipeline_diarize(str(wav_yolu))
    with open("output.rttm", "w") as f:
        diarization.write_rttm(f)
    print("Konuşmacılar belirlendi.")
    return diarization


def transkripsiyon_yap(wav_yolu, diarization, dil, yuz_cizelgesi):
    print("\n[3/6] Transkripsiyon + ses tonu + yüz ifadesi analizi başlatılıyor...")
    ses = AudioSegment.from_wav(str(wav_yolu))
    transkript = []

    for tur, _, konusmaci in diarization.itertracks(yield_label=True):
        bas, bit = float(tur.start), float(tur.end)
        seg = ses[int(bas * 1000): int(bit * 1000)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            seg.export(tf.name, format="wav")
            gecici = tf.name

        # --- Transkripsiyon ---
        try:
            tokenizer = processor_whisper.tokenizer
            lang_id = None
            if hasattr(tokenizer, "lang_code_to_id") and dil in tokenizer.lang_code_to_id:
                lang_id = tokenizer.lang_code_to_id[dil]
            gen_kwargs = {"forced_bos_token_id": int(lang_id)} if lang_id else {}
            sonuc = transcriber(gecici, generate_kwargs=gen_kwargs)
            metin = sonuc.get("text", "").strip()
        except Exception as e:
            metin = f"[Transkripsiyon hatasi: {e}]"

        # --- Ses Tonu ---
        ses_tonu = "N/A"
        try:
            ytmp, srtmp = librosa.load(gecici, sr=16000)
            inputs = extractor(ytmp, sampling_rate=srtmp, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model_emotion(**inputs).logits
                tahmin = torch.argmax(logits, dim=1).item()
                ses_tonu = model_emotion.config.id2label[tahmin]
        except Exception:
            pass

        # --- Yüz ifadesi (segment ortasına göre) ---
        yuz_ifadesi = en_yakin_yuz_ifadesi(yuz_cizelgesi, bas, bit)

        transkript.append({
            "konusmaci": konusmaci,
            "bas": round(bas, 2),
            "bit": round(bit, 2),
            "metin": metin,
            "ses_tonu": ses_tonu,
            "yuz_ifadesi": yuz_ifadesi
        })

        os.remove(gecici)

    print("[4/6] Transkripsiyon + analiz tamamlandı.")
    return transkript


def json_kaydet(veri):
    with open(JSON_CIKTI, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)
    print(f"[5/6] JSON kaydedildi: {JSON_CIKTI}")


def txt_kaydet(veri):
    with open(TXT_CIKTI, "w", encoding="utf-8") as f:
        for s in veri:
            f.write(f"[{s['konusmaci']}] {s['metin']} (Ses tonu: {s['ses_tonu']}, Yüz ifadesi: {s['yuz_ifadesi']})\n\n")
    print(f"[6/6] TXT kaydedildi: {TXT_CIKTI}")


def calistir(dil="tr"):
    print("\nAkıllı Mülakat Analizi Başlatılıyor...")
    yuz_cizelgesi = yuz_ifadesi_analiz(VIDEO_DOSYASI)
    ses_cikar(VIDEO_DOSYASI, WAV_DOSYASI)
    diarization = konusmaci_ayir(WAV_DOSYASI)
    transkript = transkripsiyon_yap(WAV_DOSYASI, diarization, dil, yuz_cizelgesi)
    json_kaydet(transkript)
    txt_kaydet(transkript)
    print("\nAnaliz başarıyla tamamlandı.")


def main():
    print("\nMülakat dilini seçin:")
    print("1) Türkçe")
    print("2) İngilizce")
    secim = input("Seçim (1/2): ").strip()
    dil = "tr" if secim == "1" else "en"
    calistir(dil)


if __name__ == "__main__":
    main()
