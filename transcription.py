"""Transkripsiyon ve segment analizleri."""

import os
import tempfile
from collections import Counter

import librosa
import numpy as np
from pydub import AudioSegment

import audio_models
import audio_processing
import config


def transkripsiyon_yap(wav_yolu, diarization, dil, yuz_cizelgesi):
    """Transkripsiyon yapar, dolgu, duraksama, ses tonu ve yüz ifadelerini ekler."""
    ses = AudioSegment.from_wav(str(wav_yolu))
    transkript = []
    duraksamalar = audio_processing.duraksama_tespit(wav_yolu)
    dolgular = audio_processing.dolgu_tespit(wav_yolu)

    def segment_icindeki_dolgular(bas, bit):
        return [(s, e) for (s, e) in dolgular if s >= bas and s <= bit]

    def segment_yuz(bas, bit):
        return [y["duygu"] for y in yuz_cizelgesi if bas <= y["zaman"] <= bit]

    def segment_oncesi_duraksama(bas):
        for s, e in duraksamalar:
            if e <= bas + 0.05 and e >= bas - 3.0:
                return (s, e)
        return None

    for tur, _, konusmaci in diarization.itertracks(yield_label=True):
        bas, bit = float(tur.start), float(tur.end)
        seg = ses[int(bas * 1000): int(bit * 1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            seg.export(tf.name, format="wav")
            gecici = tf.name
        try:
            if audio_models.transcriber is None:
                raise RuntimeError("Whisper modeli yüklenemedi.")
            sonuc = audio_models.transcriber(gecici, generate_kwargs={"language": dil})
            metin = sonuc.get("text", "").strip()
        except Exception as exc:  # noqa: BLE001
            metin = f"[Transkripsiyon hatasi: {exc}]"

        durak = segment_oncesi_duraksama(bas)
        if durak and (durak[1] - durak[0]) >= 0.7:
            metin = "(duraksama) " + metin

        ses_tonu = audio_processing.hesapla_ses_tonu(gecici)

        yuzler = segment_yuz(bas, bit)
        yuz_ifadesi = Counter(yuzler).most_common(1)[0][0] if yuzler else "BELİRSİZ"

        seg_dolgu = segment_icindeki_dolgular(bas, bit)
        kelime_sayisi = len(metin.split())
        if seg_dolgu and metin.strip() and kelime_sayisi >= 2:
            seg_dolgu_sorted = sorted(
                seg_dolgu, key=lambda x: x[1] - x[0], reverse=True
            )
            chosen = seg_dolgu_sorted[: config.MAX_DOLGU_PER_SEGMENT]
            kelimeler = metin.split()
            uzunluk = max(0.0001, bit - bas)
            insert_positions = []
            for ds, de in chosen:
                rel_pos = (ds - bas) / uzunluk
                word_pos = int(rel_pos * len(kelimeler))
                insert_positions.append(min(len(kelimeler), word_pos))
            for i, pos in enumerate(sorted(insert_positions)):
                etiket = np.random.choice(
                    config.DOLGU_TR if dil == "tr" else config.DOLGU_EN
                )
                kelimeler.insert(pos + i, etiket)
            metin = " ".join(kelimeler)

        os.remove(gecici)

        transkript.append(
            {
                "konusmaci": konusmaci,
                "bas": round(bas, 3),
                "bit": round(bit, 3),
                "metin": metin,
                "ses_tonu": ses_tonu,
                "yuz_ifadesi": yuz_ifadesi,
            }
        )
    print("[4/7] Transkripsiyon + analiz tamamlandı.")
    return transkript

