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
-
LOCAL DUYGU MODEL LABEL SÖZLÜĞÜ EKLENDİ: LABEL_0, LABEL_1 vb yerine Türkçe/İngilizce okunabilir isimler gösterilecek.
Localde çalısan iyi çalısmıyor iyi sonuç üretmedi.7sesli eğittiğim.İngilizce olan türkçe de iyi sonuç vermedi
ingilizce video vericem denemesini yapalım :Sonuclar kötü çıktı.Bu yüzden bu model 7 eğitili sesle eğittik ama isteğimiz sonucu alamadık

NOT:Ses tonu için fine yuning yapıldı.
TESS VERİ SETİYLE EĞİTİLDİ BU MODEL BU MODEL İNG 7 DUYGU VERİ SETİ İÇERMESİNE RAĞMEN İSTENİLEN SONUÇ ALINAMADI.
Funi tuning sonucları:
Step	Training_Loss	Validation_Loss	  Accuracy	     F1
500	    0.920300	     0.696107	     0.864286	   0.856672
📊 Doğruluk: 0.8643 | F1 Skoru: 0.8567
"""



import os
import subprocess
import tempfile
import json
from pathlib import Path
from pydub import AudioSegment, silence
from pyannote.audio import Pipeline
import whisper
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ---------- AYARLAR ----------
VIDEO_DOSYASI = "How to prepare for your Microsoft Interview_ Virtual Interview.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"
HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large"

# --- Local duygu modeli yolu ---
LOCAL_DUYGU_MODEL = "./emre_superverb_egitimli"  # Buraya senin local model klasörünü yaz

# Çıktı dosyaları
JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

# Ses analizi parametreleri (OPTİMİZE EDİLDİ)
DURAK_MIN_MS = 400
SES_ESIK_DB = -40

# Dolgu parametreleri (güncellendi)
DOLGU_MIN_SN = 0.12
DOLGU_MAX_SN = 0.60
DOLGU_RMS_MIN = 0.005
DOLGU_RMS_MAX = 0.020
DOLGU_MERGE_GAP = 0.25
MAX_DOLGU_PER_SEGMENT = 1

DOLGU_TR = ["(ııı)", "(eee)", "(hmm)", "(şey)"]
DOLGU_EN = ["(uh)", "(um)", "(hmm)", "(like)"]

# --- Local model için label sözlükleri ---
LABEL_DICT_TR = {
    "LABEL_0": "öfke",
    "LABEL_1": "tiksinti",
    "LABEL_2": "korku",
    "LABEL_3": "mutlu",
    "LABEL_4": "nötr",
    "LABEL_5": "hoş_sürpriz",
    "LABEL_6": "üzgün"
}

LABEL_DICT_EN = {
    "LABEL_0": "angry",
    "LABEL_1": "disgust",
    "LABEL_2": "fear",
    "LABEL_3": "happy",
    "LABEL_4": "neutral",
    "LABEL_5": "pleasant_surprise",
    "LABEL_6": "sad"
}

# 🔹 Duygu modeli yükleme (local)
print("Local ses tonu modeli yükleniyor...")
try:
    extractor = AutoFeatureExtractor.from_pretrained(LOCAL_DUYGU_MODEL)
    model_emotion = AutoModelForAudioClassification.from_pretrained(LOCAL_DUYGU_MODEL)
    model_emotion.eval()
except Exception as e:
    print("Duygu modeli yüklemede hata (devam edilecek):", e)
    extractor = None
    model_emotion = None

# ---------- FONKSİYONLAR ----------

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
    print("Dolgu sesleri analiz ediliyor (optimize)...")
    y, sr = librosa.load(wav_yolu, sr=16000)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zaman = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length, n_fft=frame_length)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        voiced_mask = ~np.isnan(f0)
        f0_time = np.linspace(0, len(y)/sr, num=len(f0))
        rms_time = zaman
        f0_interp = np.interp(rms_time, f0_time, np.nan_to_num(f0, nan=0.0))
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
                dolgu.append((round(bas,3), round(bit,3), sure, float(np.mean(rms[g]))))

    if dolgu:
        temiz = []
        dolgu.sort(key=lambda x: x[0])
        cur = dolgu[0]
        for nxt in dolgu[1:]:
            if nxt[0] - cur[1] <= DOLGU_MERGE_GAP:
                cur = (cur[0], nxt[1], round(nxt[1]-cur[0],3), max(cur[3], nxt[3]))
            else:
                temiz.append(cur)
                cur = nxt
        temiz.append(cur)
        dolgu = [(b,e) for (b,e,s,r) in temiz]
    else:
        dolgu = []

    print("Dolgu sayisi bulundu (optimizasyon sonrası):", len(dolgu))
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
        seg = ses[int(bas*1000): int(bit*1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            seg.export(tf.name, format="wav")
            gecici = tf.name

        print("Islenen segment:", konusmaci, f"[{bas:.2f}-{bit:.2f}]")

        try:
            sonuc = model.transcribe(gecici, language=dil)
            metin = sonuc.get("text", "").strip()
        except Exception as e:
            metin = f"[Transkripsiyon hatasi: {e}]"

        durak = segment_oncesi_duraksama(bas)
        if durak and (durak[1]-durak[0]) >= 0.7:
            metin = "(duraksama) " + metin

        seg_dolgu = segment_icindeki_dolgular(bas, bit)
        ses_tonu = "[N/A]"

        # 🔹 Local model duygu tahmini
        if extractor is not None and model_emotion is not None:
            try:
                ytmp, srtmp = librosa.load(gecici, sr=16000)
                inputs = extractor(ytmp, sampling_rate=srtmp, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model_emotion(**inputs).logits
                    tahmin = torch.argmax(logits, dim=1).item()
                    label_key = f"LABEL_{tahmin}"
                    ses_tonu = LABEL_DICT_TR[label_key] if dil=="tr" else LABEL_DICT_EN[label_key]
            except Exception as e:
                ses_tonu = f"[Duygu tespisi hatasi: {e}]"

        kelime_sayisi = len(metin.split())
        if seg_dolgu and metin.strip() and kelime_sayisi >= 2:
            seg_dolgu_sorted = sorted(seg_dolgu, key=lambda x: x[1]-x[0], reverse=True)
            chosen = seg_dolgu_sorted[:MAX_DOLGU_PER_SEGMENT]
            kelimeler = metin.split()
            uzunluk = max(0.0001, bit-bas)
            seg_pos = []
            for (ds,de) in chosen:
                rel = (ds-bas)/uzunluk
                pos = int(rel*len(kelimeler))
                pos = max(0,min(len(kelimeler),pos))
                seg_pos.append((pos, ds,de))
            used_positions = set()
            for pos, ds,de in seg_pos:
                if pos in used_positions:
                    pos = min(len(kelimeler), pos+1)
                used_positions.add(pos)
                etiketler = DOLGU_TR if dil=="tr" else DOLGU_EN
                kelimeler.insert(pos, np.random.choice(etiketler))
            metin = " ".join(kelimeler)

        os.remove(gecici)

        transkript.append({
            "konusmaci": konusmaci,
            "bas": bas,
            "bit": bit,
            "metin": metin,
            "ses_tonu": ses_tonu,
            "duraksama": round((durak[1]-durak[0]) if durak else 0.0,3),
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
            f.write(f"[{s['konusmaci']} | {s['ses_tonu']}]: {s['metin']}\n\n")
    print("TXT kaydedildi:", dosya)


def istatistik_goster(transkript):
    print("\nMülakat Ozeti")
    toplam_dolgu = sum(s["dolgu_sayisi"] for s in transkript)
    toplam_durak = sum(s["duraksama"] for s in transkript)
    ses_tonlari = {}
    for s in transkript:
        ton = s["ses_tonu"]
        ses_tonlari[ton] = ses_tonlari.get(ton,0)+1
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
    dil = "tr" if secim=="1" else "en"
    calistir(dil)


if __name__ == "__main__":
    main()
