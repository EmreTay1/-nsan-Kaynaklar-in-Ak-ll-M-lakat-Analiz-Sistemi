#!/usr/bin/env python3
"""

- Giriş: video dosyası
- Çıkış: konuşmacı-etiketli transkript (JSON, SRT, TXT, CSV)
- Özellikler:
    - Konuşmacı ayırımı (diarization)
    - Türkçe/İngilizce Whisper transkripsiyon
    - Kullanıcı dil seçimi
    - Ses tabanlı duraksama (pause) tespiti
    - Ses tabanlı dolgu (ııı/eee/hmm/um/uh) tespiti
    - Duraksama ve dolgu etiketlerini transkripte zaman bazlı yerleştirme
    - Gelişmiş istatistik özeti

Gerekenler:
    pip install pydub librosa numpy soundfile
    ffmpeg PATH içinde olmalı
"""

import os
import subprocess
import tempfile
import json
import csv
from pathlib import Path
from datetime import timedelta
import argparse
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import whisper
import time
import numpy as np
import librosa

# ---------- AYARLAR ----------
VIDEO_DOSYASI = "video1723838072.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"
HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large"

# Çıktı dosyaları
JSON_CIKTI = "mulakat_transkript.json"
SRT_CIKTI = "mulakat_altyazi.srt"
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


class MulakatAnalizi:
    def __init__(self, video_dosya=VIDEO_DOSYASI, whisper_model=WHISPER_MODEL, dil="tr"):
        self.video_dosya = video_dosya
        self.wav_dosya = WAV_DOSYASI
        self.whisper_model = whisper_model
        self.dil = dil
        self.diarization = None
        self.transkript = []

    # ---------- Sesten WAV çıkar ----------
    def ses_cikar(self, video_yolu, cikti_wav):
        """Videodan 16kHz mono ses çıkar"""
        print("\n🎧 [1/6] Videodan ses çıkarılıyor...")
        if Path(cikti_wav).exists():
            os.remove(cikti_wav)
            print(f"⚠️ Eski ses dosyası silindi: {cikti_wav}")

        cmd = ["ffmpeg", "-y", "-i", str(video_yolu), "-ac", "1", "-ar", "16000", "-vn", str(cikti_wav)]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Ses çıkarıldı: {cikti_wav}")

    # ----------  Konuşmacı tespiti ----------
    def konusmaci_ayir(self, wav_yolu):
        """Pyannote ile konuşmacı diarization işlemi"""
        print("\n🧠 [2/6] Konuşmacılar tespit ediliyor...")
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
        self.diarization = pipeline(str(wav_yolu))
        with open("output.rttm", "w") as f:
            self.diarization.write_rttm(f)
        print("✅ Konuşmacılar belirlendi.")
        return self.diarization

    # ---------- Duraksama tespiti ----------
    def duraksama_tespit(self, wav_yolu):
        """Sessizlik (duraksama) aralıklarını bul"""
        ses = AudioSegment.from_wav(wav_yolu)
        esik = ses.dBFS + SES_ESIK_DB
        sessiz_araliklar = silence.detect_silence(ses, min_silence_len=DURAK_MIN_MS, silence_thresh=esik)
        duraksamalar = [(s / 1000.0, e / 1000.0) for s, e in sessiz_araliklar]
        print(f"🔎 {len(duraksamalar)} duraksama bulundu.")
        return duraksamalar

    # ----------  Dolgu sesi tespiti ----------
    def dolgu_tespit(self, wav_yolu):
        """Librosa RMS + pitch analiziyle ııı, eee, hmm tespiti"""
        print("🔎 Dolgu sesleri analiz ediliyor...")
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
        print(f"✅ {len(dolgu)} dolgu sesi bulundu.")
        return dolgu

    # ----------  Transkripsiyon ve zenginleştirme ----------
    def transkripsiyon_yap(self, wav_yolu, model, dil):
        """Whisper ile transkripsiyon + duraksama & dolgu etiketleri ekleme"""
        ses = AudioSegment.from_wav(str(wav_yolu))
        model = whisper.load_model(model)
        self.transkript = []

        duraksamalar = self.duraksama_tespit(wav_yolu)
        dolgular = self.dolgu_tespit(wav_yolu)

        def segment_icindeki_dolgular(bas, bit):
            return [(s, e) for (s, e) in dolgular if s >= bas and s <= bit]

        def segment_oncesi_duraksama(bas):
            for (s, e) in duraksamalar:
                if e <= bas + 0.05 and e >= bas - 3.0:
                    return (s, e)
            return None

        for tur, _, konusmaci in self.diarization.itertracks(yield_label=True):
            bas, bit = float(tur.start), float(tur.end)
            seg = ses[int(bas * 1000): int(bit * 1000)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                seg.export(tf.name, format="wav")
                gecici = tf.name

            print(f"▶️ {konusmaci} [{bas:.2f}-{bit:.2f}]...")

            try:
                sonuc = model.transcribe(gecici, language=dil)
                metin = sonuc.get("text", "").strip()
            except Exception as e:
                metin = f"[Transkripsiyon hatası: {e}]"
            os.remove(gecici)

            # Duraksama varsa başına etiket ekle
            durak = segment_oncesi_duraksama(bas)
            if durak and (durak[1] - durak[0]) >= 0.7:
                metin = "(duraksama) " + metin

            # Dolgu seslerini uygun yere yerleştir
            seg_dolgu = segment_icindeki_dolgular(bas, bit)
            if seg_dolgu and metin.strip():
                kelimeler = metin.split()
                uzunluk = max(0.0001, bit - bas)
                seg_dolgu.sort(key=lambda x: x[0])
                etiketler = DOLGU_TR if self.dil == "tr" else DOLGU_EN
                for (ds, de) in seg_dolgu:
                    rel = (ds - bas) / uzunluk
                    pos = int(rel * len(kelimeler))
                    pos = max(0, min(len(kelimeler), pos))
                    kelimeler.insert(pos, np.random.choice(etiketler))
                metin = " ".join(kelimeler)

            self.transkript.append({
                "konusmaci": konusmaci,
                "bas": bas,
                "bit": bit,
                "metin": metin,
                "duraksama": round((durak[1] - durak[0]) if durak else 0.0, 3),
                "dolgu_sayisi": len(seg_dolgu)
            })

        print("✅ Transkripsiyon tamamlandı ve metin zenginleştirildi.")
        return self.transkript

    # ----------  Çıktı kayıtları ----------
    def json_kaydet(self, dosya=JSON_CIKTI):
        with open(dosya, "w", encoding="utf-8") as f:
            json.dump(self.transkript, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON kaydedildi: {dosya}")

    def txt_kaydet(self, dosya=TXT_CIKTI):
        with open(dosya, "w", encoding="utf-8") as f:
            for s in self.transkript:
                f.write(f"[{s['konusmaci']}]: {s['metin']}\n\n")
        print(f"💾 TXT kaydedildi: {dosya}")

    # ----------  Özet ----------
    def istatistik_goster(self):
        print("\n📊 Mülakat Özeti")
        toplam_dolgu = sum(s["dolgu_sayisi"] for s in self.transkript)
        toplam_durak = sum(s["duraksama"] for s in self.transkript)
        print(f"Toplam duraksama süresi: {toplam_durak:.2f} sn")
        print(f"Toplam dolgu sayısı: {toplam_dolgu}")

    # ---------- Ana çalışma ----------
    def calistir(self):
        print("🎯 Akıllı Mülakat Analizi Başlatılıyor...")
        self.ses_cikar(self.video_dosya, self.wav_dosya)
        self.konusmaci_ayir(self.wav_dosya)
        self.transkripsiyon_yap(self.wav_dosya, self.whisper_model, self.dil)
        self.json_kaydet()
        self.txt_kaydet()
        self.istatistik_goster()
        print("\n✅ Analiz tamamlandı.")


def main():
    print("\n🌐 Mülakat dili seçin:")
    print("1️⃣ Türkçe\n2️⃣ İngilizce")
    dil_secim = input("Seçim (1/2): ").strip()
    dil = "tr" if dil_secim == "1" else "en"

    analiz = MulakatAnalizi(dil=dil)
    analiz.calistir()


if __name__ == "__main__":
    main()
