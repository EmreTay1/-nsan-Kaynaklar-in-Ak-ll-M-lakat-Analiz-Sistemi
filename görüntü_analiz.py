import cv2
from collections import Counter
from deepface import DeepFace

# Yüz ifadelerini Türkçeye çevirmek için sözlük
DUYGU_SOZLUGU = {
    'angry': 'ÖFKELİ',
    'disgust': 'TİKSİNMİŞ',
    'fear': 'KORKMUŞ',
    'happy': 'MUTLU',
    'sad': 'ÜZGÜN',
    'surprise': 'ŞAŞKIN',
    'neutral': 'NÖTR'
}


def yuz_ifadesi_analiz(video_yolu, saniye_araligi=0.5):
    """
    Video üzerinden yüz ifadelerini tespit eder ve zaman damgalarıyla döndürür.

    Args:
        video_yolu (str): Video dosyası yolu
        saniye_araligi (float): Her kaç saniyede bir kare analiz edilecek

    Returns:
        list: [{'zaman': <saniye>, 'duygu': <yüz ifadesi>}, ...]
    """
    print("[Yüz ifadeleri analizi başlatılıyor...]")
    video = cv2.VideoCapture(video_yolu)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * saniye_araligi)  # kaç karede bir analiz

    yuz_cizelgesi = []
    frame_num = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Belirlenen kare aralığında analiz
        if frame_num % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                # DeepFace bazen liste döndürüyor
                dominant_emotion = None
                if isinstance(analysis, list) and len(analysis) > 0:
                    dominant_emotion = analysis[0]['dominant_emotion']
                elif isinstance(analysis, dict):
                    dominant_emotion = analysis.get('dominant_emotion', None)

                if dominant_emotion:
                    dominant_tr = DUYGU_SOZLUGU.get(dominant_emotion, dominant_emotion.upper())
                    timestamp = frame_num / fps
                    yuz_cizelgesi.append({'zaman': timestamp, 'duygu': dominant_tr})
            except Exception as e:
                # Hata olursa atla
                pass

        frame_num += 1

    video.release()
    print("[Yüz ifadeleri analizi tamamlandı.]")
    return yuz_cizelgesi


# ----------------- Örnek Kullanım -----------------
video_path = "video1723838072.mp4"
yuz_veri = yuz_ifadesi_analiz(video_path, saniye_araligi=0.5)

# Çıktıyı yazdır
for y in yuz_veri:
    print(f"Saniye: {y['zaman']:.2f} | Yüz ifadesi: {y['duygu']}")
