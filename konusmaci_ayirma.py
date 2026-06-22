#!/usr/bin/env python3
"""

- Giriş: video dosyası
- Çıkış: konuşmacı-etiketli transkript (JSON, SRT, TXT, CSV formatlarında)
- Özellikler: konuşmacı diarization, Türkçe transkripsiyon, çoklu çıktı formatı
"""

import os
import subprocess
import tempfile
import json
import csv
from pathlib import Path
from datetime import timedelta
import argparse

# Ses işleme kütüphaneleri
from pydub import AudioSegment
import soundfile as sf

# AI modelleri
from pyannote.audio import Pipeline
import whisper

# ---------- KONFIGÜRASYON ----------
VIDEO_FILE = "How to prepare for your Microsoft Interview_ Virtual Interview.mp4"
WAV_FILE = "gecici_16k_mono.wav"
HF_TOKEN = "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"  # ← BURAYA MANUEL TOKEN'I YAZ
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large"  # tiny/base/small/medium/large
OUTPUT_JSON = "mulakat_transkript.json"
OUTPUT_SRT = "mulakat_altyazi.srt"
OUTPUT_TXT = "mulakat_metin.txt"
OUTPUT_CSV = "mulakat_detay.csv"


# -----------------------------

class MulakatAnalizi:
    def __init__(self, video_file=VIDEO_FILE, whisper_model=WHISPER_MODEL):
        self.video_file = video_file
        self.wav_file = WAV_FILE
        self.whisper_model = whisper_model
        self.diarization = None
        self.transcript = []

    def ses_cikart(self, video_path, out_wav):
        """Video dosyasından 16kHz mono WAV çıkar"""
        print(f"[+] Video dosyasından ses çıkarılıyor: {video_path}")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[+] Ses dosyası hazır: {out_wav}")
        except subprocess.CalledProcessError as e:
            print(f"[-] FFmpeg hatası: {e}")
            raise

    def diarization_yap(self, wav_path):
        """Konuşmacı diarization (kim, ne zaman konuşuyor)"""
        if not HF_TOKEN or HF_TOKEN.startswith("hf_xxx"):
            raise RuntimeError(
                "HF_TOKEN manuel olarak kodda tanımlanmalı! Lütfen token'ı HF_TOKEN değişkenine yazın."
            )

        print("[+] Pyannote pipeline yükleniyor (modeller indirilecek)...")
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)

        print("[+] Diarization çalıştırılıyor (bu işlem zaman alabilir)...")
        self.diarization = pipeline(str(wav_path))

        # RTTM dosyasını kaydet
        with open("output.rttm", "w") as f:
            self.diarization.write_rttm(f)
        print("[+] RTTM dosyası kaydedildi: output.rttm")

        return self.diarization

    def transkripsiyon_yap(self, wav_path):
        """Her segment için Whisper ile transkripsiyon"""
        print("[+] Ses dosyası belleğe yükleniyor...")
        audio = AudioSegment.from_wav(str(wav_path))

        print(f"[+] Whisper modeli yükleniyor: {self.whisper_model}")
        whisper_model = whisper.load_model(self.whisper_model)

        print("[+] Segmentler transkripsiyon ediliyor...")
        self.transcript = []

        for turn, _, speaker in self.diarization.itertracks(yield_label=True):
            start, end = float(turn.start), float(turn.end)

            # Segment çıkar
            seg = audio[int(start * 1000): int(end * 1000)]

            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                seg.export(tf.name, format="wav")
                tmpname = tf.name

            # Transkripsiyon yap
            try:
                res = whisper_model.transcribe(tmpname, language="tr")
                text = res.get("text", "").strip()
            except Exception as e:
                text = f"[TRANSKRİPSİYON HATASI: {e}]"

            # Geçici dosyayı sil
            os.remove(tmpname)

            # Sonucu kaydet
            segment_data = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text
            }
            self.transcript.append(segment_data)

            # İlerleme göster
            print(f"{speaker} [{start:.2f}-{end:.2f}]: {text[:80]}...")

        return self.transcript

    def json_kaydet(self, filename=OUTPUT_JSON):
        """JSON formatında kaydet"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.transcript, f, ensure_ascii=False, indent=2)
        print(f"[+] JSON kaydedildi: {filename}")

    def srt_kaydet(self, filename=OUTPUT_SRT):
        """SRT altyazı formatında kaydet"""

        def zaman_formatla(saniye):
            td = timedelta(seconds=saniye)
            hours, remainder = divmod(td.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            milliseconds = int((seconds % 1) * 1000)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

        with open(filename, "w", encoding="utf-8") as f:
            for i, segment in enumerate(self.transcript, 1):
                start_time = zaman_formatla(segment["start"])
                end_time = zaman_formatla(segment["end"])
                speaker = segment["speaker"]
                text = segment["text"]

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{speaker}] {text}\n\n")

        print(f"[+] SRT altyazı kaydedildi: {filename}")

    def txt_kaydet(self, filename=OUTPUT_TXT):
        """Düz metin formatında kaydet"""
        with open(filename, "w", encoding="utf-8") as f:
            for segment in self.transcript:
                speaker = segment["speaker"]
                text = segment["text"]
                f.write(f"[{speaker}]: {text}\n\n")

        print(f"[+] Metin dosyası kaydedildi: {filename}")

    def csv_kaydet(self, filename=OUTPUT_CSV):
        """CSV formatında detaylı kaydet"""
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Konuşmacı", "Başlangıç (s)", "Bitiş (s)", "Süre (s)", "Metin"])

            for segment in self.transcript:
                writer.writerow([
                    segment["speaker"],
                    f"{segment['start']:.2f}",
                    f"{segment['end']:.2f}",
                    f"{segment['duration']:.2f}",
                    segment["text"]
                ])

        print(f"[+] CSV dosyası kaydedildi: {filename}")

    def istatistik_goster(self):
        """Analiz istatistiklerini göster"""
        if not self.transcript:
            print("[-] Henüz transkript yok.")
            return

        # Konuşmacı istatistikleri
        speakers = {}
        total_duration = 0

        for segment in self.transcript:
            speaker = segment["speaker"]
            duration = segment["duration"]

            if speaker not in speakers:
                speakers[speaker] = {"duration": 0, "segments": 0, "words": 0}

            speakers[speaker]["duration"] += duration
            speakers[speaker]["segments"] += 1
            speakers[speaker]["words"] += len(segment["text"].split())
            total_duration += duration

        print("\n" + "=" * 50)
        print("MÜLAKAT ANALİZ İSTATİSTİKLERİ")
        print("=" * 50)
        print(f"Toplam süre: {total_duration:.2f} saniye ({total_duration / 60:.1f} dakika)")
        print(f"Toplam segment: {len(self.transcript)}")
        print(f"Konuşmacı sayısı: {len(speakers)}")
        print("\nKonuşmacı detayları:")

        for speaker, stats in speakers.items():
            percentage = (stats["duration"] / total_duration) * 100
            print(f"  {speaker}:")
            print(f"    - Süre: {stats['duration']:.2f}s (%{percentage:.1f})")
            print(f"    - Segment: {stats['segments']}")
            print(f"    - Kelime: {stats['words']}")

    def calistir(self):
        """Ana pipeline'ı çalıştır"""
        print("🎯 Akıllı Mülakat Analizi Başlıyor...")
        print("=" * 50)

        # 1) Ses çıkar
        if Path(self.wav_file).exists():
            print(f"[+] Mevcut WAV dosyası bulundu: {self.wav_file}")
        elif Path(self.video_file).exists():
            self.ses_cikart(self.video_file, self.wav_file)
        else:
            raise FileNotFoundError(f"Video dosyası bulunamadı: {self.video_file}")

        # 2) Diarization
        self.diarization_yap(self.wav_file)

        # 3) Transkripsiyon
        self.transkripsiyon_yap(self.wav_file)

        # 4) Tüm formatlarda kaydet
        self.json_kaydet()
        self.srt_kaydet()
        self.txt_kaydet()
        self.csv_kaydet()

        # 5) İstatistikleri göster
        self.istatistik_goster()

        print("\n✅ Analiz tamamlandı!")
        print(f"📁 Çıktı dosyaları:")
        print(f"   - JSON: {OUTPUT_JSON}")
        print(f"   - SRT: {OUTPUT_SRT}")
        print(f"   - TXT: {OUTPUT_TXT}")
        print(f"   - CSV: {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Akıllı Mülakat Analizi")
    parser.add_argument("--video", default=VIDEO_FILE, help="Video dosyası yolu")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model boyutu")
    parser.add_argument("--output-dir", help="Çıktı dizini")

    args = parser.parse_args()

    # Çıktı dizini ayarla
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    # Analizi başlat
    analyzer = MulakatAnalizi(video_file=args.video, whisper_model=args.model)
    analyzer.calistir()


if __name__ == "__main__":
    main()