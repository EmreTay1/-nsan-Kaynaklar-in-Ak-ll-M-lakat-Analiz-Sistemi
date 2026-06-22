"""Video + ses biyometrik analiz ana akışı."""

import uuid

import config
from audio_processing import konusmaci_ayir, ses_cikar
from db_log import log_imha_kaydi
from io_ops import (
    gecici_dosyalari_temizle,
    json_kaydet,
    kalici_dosyalari_imha_et,
    txt_kaydet,
)
from transcription import transkripsiyon_yap
from video_processing import yuz_ifadesi_analiz
from llm_analysis import transkript_jsondan_docx_olustur, ik_analyze_interview


def biyometrik_calistir(dil="tr", video_path=None, islem_uuid=None) -> str:
    """
    video_path: GUI'den seçilen video yolu. None ise config.VIDEO_DOSYASI kullanılır.
    islem_uuid: dışarıdan verilebilir; yoksa otomatik üretir. Dönüşte UUID döner.
    """
    islem_uuid = islem_uuid or str(uuid.uuid4())
    video_path = video_path or config.VIDEO_DOSYASI

    print(f"Akıllı Mülakat Analizi Başlatılıyor... (IslemUUID={islem_uuid})")

    # 1. Aşama: Veri Çıkarımı
    ses_cikar(video_path, config.WAV_DOSYASI)
    diarization = konusmaci_ayir(config.WAV_DOSYASI)
    yuz_cizelgesi = yuz_ifadesi_analiz(video_path)
    transkript = transkripsiyon_yap(config.WAV_DOSYASI, diarization, dil, yuz_cizelgesi)

    # 2. Aşama: Kayıt
    json_kaydet(transkript)
    txt_kaydet(transkript)
    print("[7/7] Ham transkript kaydedildi (JSON ve TXT).")

    # 3. Aşama: Ara Rapor (Transkript DOCX)
    print("[8/9] Anonimleştirme yapılıyor ve Ara DOCX oluşturuluyor...")
    transkript_jsondan_docx_olustur(
        json_yolu=config.JSON_CIKTI,
        docx_yolu=config.INPUT_DOCX_FILE,
        anonimlestir=True,
    )

    # 4. Aşama: LLM Analizi (Nihai Rapor)
    print("[9/9] LLM ile İK Analiz Raporu oluşturuluyor...")
    ik_analyze_interview()
    print("Analiz raporu oluşturuldu.")

    # 5. Aşama: Temizlik ve İmha (ARTIK EN SONDA)
    print("\n[Son] Temizlik ve imha başlıyor...")
    gecici_durum = gecici_dosyalari_temizle()
    orjinal_durum, transkript_durum = kalici_dosyalari_imha_et(video_path)

    # 6. Aşama: Loglama
    log_imha_kaydi(
        islem_uuid=islem_uuid,
        orjinal_video_durumu=orjinal_durum[:150],
        gecici_dosya_durumu=gecici_durum[:150],
        transkript_durumu=transkript_durum[:150],
        genel_durum="Tamamlandı (Analiz Raporu Tutuldu)",
    )
    return islem_uuid