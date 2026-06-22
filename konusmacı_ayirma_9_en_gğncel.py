#!/usr/bin/env python3
"""
Akıllı Mülakat Analizi + Ses Tonu (Duygu) Tespiti

GÜNCELLEME NOTLARI (EN BAŞTA):
- DOLGU OPTİMİZASYONU:
  1) DOLGU_RMS aralığı daraltıldı: daha sessiz/kusurlu sinyaller dikkate alınmaz.
  2) DOLGU süre aralığı daraltıldı: 0.12 - 0.6 s (çok kısa/çok uzunlar elendi).
  3) Ardışık ve çok yakın dolgular 250 ms eşik ile birleştiriliyor.
  4) Segment başına maksimum 1 dolgu konulacak şekilde sınırlandı.
  5) Çok kısa transkriptlerde (kelime sayısı < 2) dolgu eklenmiyor.
  6) Dolgu tespitinde hem RMS hem f0 (voiced) kontrolü birlikte kullanılıyor.
- DOLGU YERLEŞTİRME:
  - Relatif pozisyon hesaplanırken kelime sayısı 0 hatasına karşı koruma.
  - Aynı segmentte çok sayıda dolgu varsa yalnızca en belirgin (uzun/orta enerjili) seçiliyor.
  5) Çok kısa transkriptlerde (kelime sayısı < 2) dolgu eklenmiyor.
"""

# --------------------------------------------
# GÜNCELLEME NOTU (SON):
# - Whisper modülü tamamen kaldırıldı.
# - Hugging Face "openai/whisper-large-v3" pipeline ile entegre edildi.
# - GPU varsa otomatik kullanım, CPU’da da çalışacak.
# - Türkçe dil parametresi segment bazında doğru geçiliyor.
# - Transkripsiyon doğruluğu arttı ve İngilizce/Türkçe karışıklıkları düzeltildi.
# --------------------------------------------

#!/usr/bin/env python3
"""
Akıllı Mülakat Analizi + Ses Tonu (Duygu) Tespiti

Güncelleme: Whisper pipeline, Hugging Face yeni API'siyle (AutoModelForSpeechSeq2Seq + AutoProcessor) uyumlu hale getirildi.
"""

# !/usr/bin/env python3
"""
Akıllı Mülakat Analizi + Ses Tonu (Duygu) Tespiti (Whisper large-v3 Güncellemesi)

GÜNCELLEME NOTLARI:
- WHISPER MODELİ: OpenAI'nin orijinal 'whisper' kütüphanesi yerine, Hugging Face
  'transformers' kütüphanesi kullanılarak daha güncel ve performanslı olan
  `openai/whisper-large-v3` modeli entegre edilmiştir.
- PERFORMANS: Model yüklemesi, GPU (CUDA) varsa otomatik olarak GPU'yu kullanacak
  ve daha hızlı işlem için float16 veri tipini tercih edecek şekilde ayarlandı.
- KÜTÜPHANE BAĞIMLILIKLARI: 'transformers', 'torch', ve 'accelerate'
  kütüphanelerinin kurulu olması gerekmektedir.
- DOLGU OPTİMİZASYONU: Önceki optimizasyonlar korunmuştur.
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
# --- Hugging Face Whisper için gerekli kütüphaneler ---
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ---------- AYARLAR ----------
VIDEO_DOSYASI = "video1723838072.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"
HF_TOKEN = "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"  # Pyannote için gerekli
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
# --- Whisper Modeli Güncellendi ---
#ing iyi ama türkçe için idare eder "openai/whisper-large-v3"
#tr için fine tuning yapılmış "emre/whisper-medium-turkish-2" çalısmadı?
#deneme :ysdede/Phi-4-mm-inst-asr-turkish
WHISPER_MODEL_ID ="openai/whisper-large-v3"

# --- Duygu Modeli ---
DUYGU_MODEL = "superb/wav2vec2-large-superb-er"

# --- Çıktı dosyaları ---
JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

# --- Ses analizi parametreleri ---
DURAK_MIN_MS = 400
SES_ESIK_DB = -40

# --- Dolgu parametreleri ---
DOLGU_MIN_SN = 0.12
DOLGU_MAX_SN = 0.60
DOLGU_RMS_MIN = 0.005
DOLGU_RMS_MAX = 0.020
DOLGU_MERGE_GAP = 0.25
MAX_DOLGU_PER_SEGMENT = 1
DOLGU_TR = ["(ııı)", "(eee)", "(hmm)", "(şey)"]
DOLGU_EN = ["(uh)", "(um)", "(hmm)", "(like)"]

# 🔹 Duygu modeli yükleme
print("Ses tonu modeli yükleniyor...")
try:
    extractor = AutoFeatureExtractor.from_pretrained(DUYGU_MODEL)
    model_emotion = AutoModelForAudioClassification.from_pretrained(DUYGU_MODEL)
    model_emotion.eval()
except Exception as e:
    print(f"Duygu modeli yüklemede hata (devam edilecek): {e}")
    extractor = None
    model_emotion = None

# 🔹 Whisper modeli yükleme (Hugging Face'den)
print(f"Whisper modeli yükleniyor: {WHISPER_MODEL_ID}...")
try:
    # Cihazı belirle (varsa GPU kullan)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Modeli ve işlemciyi yükle
    model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model_whisper.to(device)
    processor_whisper = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    # Transkripsiyon için pipeline oluştur
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
    print("Whisper modeli başarıyla yüklendi.")
except Exception as e:
    print(f"Whisper modeli yüklenirken bir hata oluştu: {e}")
    transcriber = None


# ---------- FONKSİYONLAR ----------

def ses_cikar(video_yolu, cikti_wav):
    print("\n[1/6] Videodan ses çıkarılıyor...")
    if Path(cikti_wav).exists():
        os.remove(cikti_wav)
        print(f"Eski ses dosyası silindi: {cikti_wav}")
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


def duraksama_tespit(wav_yolu):
    ses = AudioSegment.from_wav(wav_yolu)
    esik = ses.dBFS + SES_ESIK_DB
    sessiz_araliklar = silence.detect_silence(ses, min_silence_len=DURAK_MIN_MS, silence_thresh=esik)
    duraksamalar = [(s / 1000.0, e / 1000.0) for s, e in sessiz_araliklar]
    print(f"Duraksama sayisi bulundu: {len(duraksamalar)}")
    return duraksamalar


def dolgu_tespit(wav_yolu):
    print("Dolgu sesleri analiz ediliyor (optimize)...")
    y, sr = librosa.load(wav_yolu, sr=16000)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zaman = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_time = np.linspace(0, len(y) / sr, num=len(f0))
        f0_interp = np.interp(zaman, f0_time, np.nan_to_num(f0, nan=0.0))
        voiced_frame_mask = f0_interp > 0.0
    except Exception:
        voiced_frame_mask = np.zeros_like(rms, dtype=bool)

    rms_mask = (rms >= DOLGU_RMS_MIN) & (rms <= DOLGU_RMS_MAX)
    maske = rms_mask & voiced_frame_mask

    dolgu = []
    if maske.any():
        idx = np.where(maske)[0]
        gruplar = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in gruplar:
            bas, bit = zaman[g[0]], zaman[g[-1]]
            sure = bit - bas
            if DOLGU_MIN_SN <= sure <= DOLGU_MAX_SN:
                dolgu.append((round(bas, 3), round(bit, 3), sure, float(np.mean(rms[g]))))

    if dolgu:
        temiz = []
        dolgu.sort(key=lambda x: x[0])
        cur = dolgu[0]
        for nxt in dolgu[1:]:
            if nxt[0] - cur[1] <= DOLGU_MERGE_GAP:
                cur = (cur[0], nxt[1], round(nxt[1] - cur[0], 3), max(cur[3], nxt[3]))
            else:
                temiz.append(cur)
                cur = nxt
        temiz.append(cur)
        dolgu = [(b, e) for b, e, s, r in temiz]

    print(f"Dolgu sayisi bulundu (optimizasyon sonrası): {len(dolgu)}")
    return dolgu


def transkripsiyon_yap(wav_yolu, diarization, dil):
    print("\n[3/6] Transkripsiyon ve analiz başlatılıyor...")
    ses = AudioSegment.from_wav(str(wav_yolu))
    transkript = []

    duraksamalar = duraksama_tespit(wav_yolu)
    dolgular = dolgu_tespit(wav_yolu)

    def segment_icindeki_dolgular(bas, bit):
        return [(s, e) for (s, e) in dolgular if s >= bas and s <= bit]

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

        print(f"İşlenen segment: {konusmaci} [{bas:.2f}-{bit:.2f}]")

        try:
            # --- YENİ TRANSKRİPSİYON YÖNTEMİ ---
            if transcriber is None:
                raise Exception("Whisper modeli yüklenemedi, transkripsiyon atlanıyor.")

            sonuc = transcriber(gecici, generate_kwargs={"language": dil})
            metin = sonuc.get("text", "").strip()
            # ------------------------------------
        except Exception as e:
            metin = f"[Transkripsiyon hatasi: {e}]"

        # Duraksama kontrolü
        durak = segment_oncesi_duraksama(bas)
        if durak and (durak[1] - durak[0]) >= 0.7:
            metin = "(duraksama) " + metin

        # Ses tonu tespiti
        ses_tonu = "[N/A]"
        if extractor and model_emotion:
            try:
                ytmp, srtmp = librosa.load(gecici, sr=16000)
                inputs = extractor(ytmp, sampling_rate=srtmp, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model_emotion(**inputs).logits
                    tahmin = torch.argmax(logits, dim=1).item()
                    ses_tonu = model_emotion.config.id2label[tahmin]
            except Exception as e:
                ses_tonu = f"[Duygu tespisi hatasi: {e}]"

        # Dolgu yerleştirme
        seg_dolgu = segment_icindeki_dolgular(bas, bit)
        kelime_sayisi = len(metin.split())
        if seg_dolgu and metin.strip() and kelime_sayisi >= 2:
            seg_dolgu_sorted = sorted(seg_dolgu, key=lambda x: x[1] - x[0], reverse=True)
            chosen = seg_dolgu_sorted[:MAX_DOLGU_PER_SEGMENT]
            kelimeler = metin.split()
            uzunluk = max(0.0001, bit - bas)

            insert_positions = []
            for ds, de in chosen:
                rel_pos = (ds - bas) / uzunluk
                word_pos = int(rel_pos * len(kelimeler))
                insert_positions.append(min(len(kelimeler), word_pos))

            # Aynı yere eklemeyi önlemek için pozisyonları ayarla
            for i, pos in enumerate(sorted(insert_positions)):
                etiket = np.random.choice(DOLGU_TR if dil == "tr" else DOLGU_EN)
                kelimeler.insert(pos + i, etiket)

            metin = " ".join(kelimeler)

        os.remove(gecici)

        transkript.append({
            "konusmaci": konusmaci,
            "bas": round(bas, 3),
            "bit": round(bit, 3),
            "metin": metin,
            "ses_tonu": ses_tonu,
            "duraksama": round((durak[1] - durak[0]) if durak else 0.0, 3),
            "dolgu_sayisi": len(seg_dolgu)
        })

    print("\n[4/6] Transkripsiyon + ses tonu analizi tamamlandi.")
    return transkript


def json_kaydet(veri, dosya=JSON_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=2)
    print(f"\n[5/6] JSON kaydedildi: {dosya}")


def txt_kaydet(veri, dosya=TXT_CIKTI):
    with open(dosya, "w", encoding="utf-8") as f:
        for s in veri:
            f.write(f"[{s['konusmaci']} | {s['ses_tonu']}]: {s['metin']}\n\n")
    print(f"[5/6] TXT kaydedildi: {dosya}")


def istatistik_goster(transkript):
    print("\n[6/6] Mülakat Özeti")
    toplam_dolgu = sum(s["dolgu_sayisi"] for s in transkript)
    toplam_durak = sum(s["duraksama"] for s in transkript)
    ses_tonlari = {}
    for s in transkript:
        ton = s["ses_tonu"]
        ses_tonlari[ton] = ses_tonlari.get(ton, 0) + 1

    print("---" * 10)
    print(f"Toplam duraksama süresi: {toplam_durak:.2f} sn")
    print(f"Tespit edilen toplam dolgu sayısı: {toplam_dolgu}")
    print("Ses tonu dağılımı:", ses_tonlari)
    print("---" * 10)


def calistir(dil="tr"):
    print("\nAkıllı Mülakat Analizi Başlatılıyor...")
    ses_cikar(VIDEO_DOSYASI, WAV_DOSYASI)
    diarization = konusmaci_ayir(WAV_DOSYASI)
    transkript = transkripsiyon_yap(WAV_DOSYASI, diarization, dil)
    json_kaydet(transkript)
    txt_kaydet(transkript)
    istatistik_goster(transkript)
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