"""Ses işleme yardımcıları."""

import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline as PyannotePipeline
import torch

import config
import audio_models


def ses_cikar(video_yolu, cikti_wav):
    """Videodan ses çıkarır (16 kHz mono WAV)."""
    print("[1/7] Videodan ses çıkarılıyor...")
    if Path(cikti_wav).exists():
        os.remove(cikti_wav)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_yolu),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(cikti_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Ses çıkarıldı: {cikti_wav}")


def konusmaci_ayir(wav_yolu):
    """Pyannote ile konuşmacıları tespit eder."""
    print("[2/7] Konuşmacılar tespit ediliyor...")
    pipeline_diarize = PyannotePipeline.from_pretrained(
        config.PYANNOTE_MODEL, use_auth_token=config.HF_TOKEN
    )
    diarization = pipeline_diarize(str(wav_yolu))
    return diarization


def duraksama_tespit(wav_yolu):
    """Duraksama / sessizlik tespiti."""
    ses = AudioSegment.from_wav(wav_yolu)
    esik = ses.dBFS + config.SES_ESIK_DB
    sessiz_araliklar = silence.detect_silence(
        ses, min_silence_len=config.DURAK_MIN_MS, silence_thresh=esik
    )
    return [(s / 1000.0, e / 1000.0) for s, e in sessiz_araliklar]


def dolgu_tespit(wav_yolu):
    """Dolgu seslerini tespit eder (optimize)."""
    y, sr = librosa.load(wav_yolu, sr=16000)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zaman = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length
    )
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_time = np.linspace(0, len(y) / sr, num=len(f0))
        f0_interp = np.interp(zaman, f0_time, np.nan_to_num(f0, nan=0.0))
        voiced_frame_mask = f0_interp > 0.0
    except Exception:  # noqa: BLE001
        voiced_frame_mask = np.zeros_like(rms, dtype=bool)
    rms_mask = (rms >= config.DOLGU_RMS_MIN) & (rms <= config.DOLGU_RMS_MAX)
    maske = rms_mask & voiced_frame_mask
    dolgu = []
    if maske.any():
        idx = np.where(maske)[0]
        gruplar = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in gruplar:
            bas, bit = zaman[g[0]], zaman[g[-1]]
            sure = bit - bas
            if config.DOLGU_MIN_SN <= sure <= config.DOLGU_MAX_SN:
                dolgu.append((round(bas, 3), round(bit, 3)))
    return dolgu


def hesapla_ses_tonu(gecici_wav):
    """Tek segment için ses tonunu döndürür."""
    if audio_models.extractor is None or audio_models.model_emotion is None:
        return "[N/A]"
    try:
        ytmp, srtmp = librosa.load(gecici_wav, sr=16000)
        inputs = audio_models.extractor(
            ytmp, sampling_rate=srtmp, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = audio_models.model_emotion(**inputs).logits
            tahmin = torch.argmax(logits, dim=1).item()
            return audio_models.model_emotion.config.id2label[tahmin]
    except Exception:  # noqa: BLE001
        return "[HATA]"

