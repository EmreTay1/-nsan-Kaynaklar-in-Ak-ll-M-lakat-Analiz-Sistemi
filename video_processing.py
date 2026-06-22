"""Görüntü/yüz ifadesi analizi yardımcıları."""

import cv2
from deepface import DeepFace

import config


def yuz_ifadesi_analiz(video_yolu, saniye_araligi=0.5):
    """Video üzerinden yüz ifadelerini tespit eder."""
    print("[3/7] Yüz ifadeleri analizi başlatılıyor...")
    video = cv2.VideoCapture(video_yolu)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * saniye_araligi)
    yuz_cizelgesi = []

    frame_num = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, silent=True
                )
                if isinstance(analysis, list) and len(analysis) > 0:
                    dominant_en = analysis[0]["dominant_emotion"]
                    dominant_tr = config.DUYGU_SOZLUGU.get(
                        dominant_en, dominant_en.upper()
                    )
                    timestamp = frame_num / fps
                    yuz_cizelgesi.append({"zaman": timestamp, "duygu": dominant_tr})
            except Exception:  # noqa: BLE001
                pass
        frame_num += 1
    video.release()
    print("Yüz ifadeleri analizi tamamlandı.")
    return yuz_cizelgesi

