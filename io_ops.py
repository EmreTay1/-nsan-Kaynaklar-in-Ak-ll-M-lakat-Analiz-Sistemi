"""Girdi/çıktı ve temizlik yardımcıları."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import config


def json_kaydet(veri, dosya=config.JSON_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)
    print(f"[5/7] JSON kaydedildi: {dosya}")


def txt_kaydet(veri, dosya=config.TXT_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        for s in veri:
            f.write(
                f"[{s['konusmaci']} | Ses tonu: {s['ses_tonu']} | "
                f"Yüz ifadesi: {s['yuz_ifadesi']}]: {s['metin']}\n\n"
            )
    print(f"[6/7] TXT kaydedildi: {dosya}")


def _guvenli_sil(dosya_yolu: str, silinenler: List[str]) -> str:
    """Dosyayı siler, durum döndürür."""
    path = Path(dosya_yolu)
    if not path.exists():
        return f"Yoktu: {path.name}"
    try:
        os.remove(path)
        silinenler.append(str(path))
        return f"Silindi: {path.name}"
    except Exception as e:  # noqa: BLE001
        return f"Silinemedi: {path.name} ({e})"


def gecici_dosyalari_temizle() -> str:
    """Analiz sonrası geçici dosyaları güvenli şekilde siler."""
    silinen_dosyalar = []
    durumlar = []

    durumlar.append(_guvenli_sil(config.WAV_DOSYASI, silinen_dosyalar))

    temp_patterns = ["temp_*.wav", "tmp_*.wav", "*_temp.wav"]
    for pattern in temp_patterns:
        for temp_file in Path(".").glob(pattern):
            durumlar.append(_guvenli_sil(temp_file, silinen_dosyalar))

    durumlar_ozet = "; ".join(durumlar) if durumlar else "Temizlenecek geçici dosya yoktu"

    if silinen_dosyalar:
        print(f" Geçici dosyalar temizlendi: {len(silinen_dosyalar)} dosya silindi")
        for dosya in silinen_dosyalar:
            print(f"   • {dosya}")
    else:
        print("  Temizlenecek geçici dosya bulunamadı.")

    return durumlar_ozet


def kalici_dosyalari_imha_et(video_path: str) -> Tuple[str, str]:
    """
    Orijinal video ve üretilen çıktı (JSON/TXT/DOCX) dosyalarını siler.
    Dönüş: (orjinal_video_durumu, transkript_durumu)
    Not: Analiz DOCX'i silmez (OUTPUT_DOCX_FILE kalır).
    """
    silinen = []
    orjinal_durum = _guvenli_sil(video_path, silinen)

    transkript_durumlar = []
    for path in [config.JSON_CIKTI, config.TXT_CIKTI, config.INPUT_DOCX_FILE]:
        transkript_durumlar.append(_guvenli_sil(path, silinen))
    transkript_durum = "; ".join(transkript_durumlar)

    if silinen:
        print(" Kalıcı dosyalar imha edildi:")
        for dosya in silinen:
            print(f"   • {dosya}")
    else:
        print("İmha edilecek kalıcı dosya bulunamadı.")

    return orjinal_durum, transkript_durum

