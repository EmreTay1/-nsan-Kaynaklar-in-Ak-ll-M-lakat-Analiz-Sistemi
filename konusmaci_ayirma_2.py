#!/usr/bin/env python3
"""
- Giriş: video dosyası
- Çıkış: konuşmacı-etiketli transkript (JSON, SRT, TXT, CSV)
- Özellikler: diarization, TR/EN Whisper transkripsiyon, kullanıcı dil seçimi, gelişmiş bilgilendirme
"""

import os
import subprocess
import tempfile
import json
import csv
from pathlib import Path
from datetime import timedelta
import argparse
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
import time

# ---------- KONFIGÜRASYON ----------
VIDEO_FILE = "video1723838072.mp4"
WAV_FILE = "gecici_16k_mono.wav"
HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large"
OUTPUT_JSON = "mulakat_transkript.json"
OUTPUT_SRT = "mulakat_altyazi.srt"
OUTPUT_TXT = "mulakat_metin.txt"
OUTPUT_CSV = "mulakat_detay.csv"


class MulakatAnalizi:
    def __init__(self, video_file=VIDEO_FILE, whisper_model=WHISPER_MODEL, language="tr"):
        self.video_file = video_file
        self.wav_file = WAV_FILE
        self.whisper_model = whisper_model
        self.language = language
        self.diarization = None
        self.transcript = []

    def ses_cikart(self, video_path, out_wav):
        """Video dosyasından 16kHz mono WAV çıkar"""
        print("\n[1/5] Video dosyasından ses çıkarılıyor...")
        if Path(out_wav).exists():
            os.remove(out_wav)
            print(f" Önceki WAV dosyası silindi: {out_wav}")

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Ses dosyası oluşturuldu: {out_wav}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg hatası: {e}")
            raise

    def diarization_yap(self, wav_path):
        """Konuşmacı diarization"""
        print("\n[2/5] Konuşmacı tespiti (diarization) başlatılıyor...")
        if not HF_TOKEN or HF_TOKEN.startswith("hf_xxx"):
            raise RuntimeError("HF_TOKEN eksik veya hatalı! HuggingFace token'ını girin.")

        start_time = time.time()
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
        self.diarization = pipeline(str(wav_path))
        with open("output.rttm", "w") as f:
            self.diarization.write_rttm(f)
        print(f"Konuşmacı segmentleri belirlendi ({time.time() - start_time:.1f} sn).")

        return self.diarization

    # ------------------ TRANSKRİPSİYON ------------------

    def transkripsiyon_tr(self, wav_path, whisper_model):
        """Türkçe transkripsiyon"""
        print("\n[3/5] Türkçe transkripsiyon başlatılıyor...")
        return self._transkripsiyon_yurut(wav_path, whisper_model, language="tr")

    def transkripsiyon_en(self, wav_path, whisper_model):
        """İngilizce transkripsiyon"""
        print("\n[3/5] English transcription started...")
        return self._transkripsiyon_yurut(wav_path, whisper_model, language="en")

    def _transkripsiyon_yurut(self, wav_path, whisper_model, language):
        """Ortak transkripsiyon işlemi"""
        audio = AudioSegment.from_wav(str(wav_path))
        whisper_model = whisper.load_model(whisper_model)
        self.transcript = []

        total_segments = sum(1 for _ in self.diarization.itertracks(yield_label=True))
        current_segment = 1

        for turn, _, speaker in self.diarization.itertracks(yield_label=True):
            start, end = float(turn.start), float(turn.end)
            seg = audio[int(start * 1000): int(end * 1000)]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                seg.export(tf.name, format="wav")
                tmpname = tf.name

            print(f" {current_segment}/{total_segments} - {speaker} [{start:.1f}-{end:.1f}]...")

            try:
                res = whisper_model.transcribe(tmpname, language=language)
                text = res.get("text", "").strip()
            except Exception as e:
                text = f"[TRANSKRİPSİYON HATASI: {e}]"

            os.remove(tmpname)

            segment_data = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text
            }
            self.transcript.append(segment_data)
            current_segment += 1

        print("Tüm segmentler transkribe edildi.")
        return self.transcript

    # ------------------ KAYDETME İŞLEMLERİ ------------------

    def json_kaydet(self, filename=OUTPUT_JSON):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.transcript, f, ensure_ascii=False, indent=2)
        print(f"JSON kaydedildi: {filename}")

    def srt_kaydet(self, filename=OUTPUT_SRT):
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
                f.write(f"{i}\n{start_time} --> {end_time}\n[{segment['speaker']}] {segment['text']}\n\n")
        print(f" SRT kaydedildi: {filename}")

    def txt_kaydet(self, filename=OUTPUT_TXT):
        with open(filename, "w", encoding="utf-8") as f:
            for segment in self.transcript:
                f.write(f"[{segment['speaker']}]: {segment['text']}\n\n")
        print(f"TXT kaydedildi: {filename}")

    def csv_kaydet(self, filename=OUTPUT_CSV):
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
        print(f"CSV kaydedildi: {filename}")

    def istatistik_goster(self):
        print("\n[5/5] Mülakat istatistikleri oluşturuluyor...")
        speakers = {}
        total_duration = 0
        for segment in self.transcript:
            sp = segment["speaker"]
            dur = segment["duration"]
            speakers.setdefault(sp, {"duration": 0, "segments": 0, "words": 0})
            speakers[sp]["duration"] += dur
            speakers[sp]["segments"] += 1
            speakers[sp]["words"] += len(segment["text"].split())
            total_duration += dur

        print("=" * 50)
        print("MÜLAKAT ANALİZ ÖZETİ")
        print("=" * 50)
        print(f"Toplam süre: {total_duration:.1f} sn ({total_duration/60:.1f} dk)")
        print(f"Toplam segment: {len(self.transcript)}")
        print(f"Konuşmacı sayısı: {len(speakers)}\n")

        for sp, st in speakers.items():
            p = (st["duration"]/total_duration)*100
            print(f"{sp}: {st['duration']:.1f}s, {st['segments']} segment, {st['words']} kelime (%{p:.1f})")

    def calistir(self):
        print(" Akıllı Mülakat Analizi Başlatılıyor...")
        print("=" * 50)

        # 1) Her zaman yeni ses çıkar
        self.ses_cikart(self.video_file, self.wav_file)

        # 2) Diarization
        self.diarization_yap(self.wav_file)

        # 3) Dil seçimi
        if self.language == "tr":
            self.transkripsiyon_tr(self.wav_file, self.whisper_model)
        else:
            self.transkripsiyon_en(self.wav_file, self.whisper_model)

        # 4) Çıktılar
        self.json_kaydet()
        self.srt_kaydet()
        self.txt_kaydet()
        self.csv_kaydet()

        # 5) İstatistikler
        self.istatistik_goster()

        print("\n Analiz tamamlandı!")


def main():
    parser = argparse.ArgumentParser(description="Akıllı Mülakat Analizi")
    parser.add_argument("--video", default=VIDEO_FILE, help="Video dosyası yolu")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model boyutu")
    parser.add_argument("--output-dir", help="Çıktı dizini")

    args = parser.parse_args()

    # Kullanıcıdan dil seçimi al
    print("\nMülakat dili nedir?")
    print("1️ Türkçe\n2️ English")
    dil_secim = input("Seçim (1/2): ").strip()
    language = "tr" if dil_secim == "1" else "en"

    # Çıktı dizini
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    analyzer = MulakatAnalizi(video_file=args.video, whisper_model=args.model, language=language)
    analyzer.calistir()


if __name__ == "__main__":
    main()
