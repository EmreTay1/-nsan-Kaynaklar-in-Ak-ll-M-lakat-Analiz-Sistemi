###############################################################
# KVKK UYARISI VE VERİ GÜVENLİĞİ:
# - Görüntü veya video verisi yalnızca yerel RAM üzerinde işlenir.
# - Frame numpy array'den PIL.Image ve tensor formatına dönüştürülür.
# - Modelin ağı yalnızca forward pass ile tahmin yapar;
#   yeni yüz bilgilerini öğrenmez veya kaydetmez.
# - İnternet kapalıyken çalıştırıldığında veri hiçbir şekilde dışarıya gönderilmez.
# - JSON çıktısı yalnızca etiket (emotion) ve olasılık (score) içerir;
#   görüntü veya kişisel veri saklanmaz.
###############################################################


# KÜTÜPHANELER
###############################################

import os
import platform
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import login
from PIL import Image
import json


###############################################
# AYARLAR
###############################################

HF_TOKEN = "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"

# Kullanmak istediğin model (manuel olarak yaz)
model_name = "sbcBI/SBC-FER"

# Video veya webcam
video_path = "video1723838072.mp4"   # Video dosyası
use_webcam = False         # True ise webcam üzerinden çalışır

# Çıktı JSON dosyası
output_json = "results_görüntü_analizi.json"

###############################################
# TÜM GÖRÜNTÜ TABANLI DUYGU ANALİZİ MODELLERİ (YORUM SATIRI)
###############################################
#Beğeninilenler
# -  "abhilash88/face-emotion-detection":istenilen çıktı değil    → ViT tabanlı, FER-2013 ile eğitilmiş 7 temel duyguyu sınıflandıran hızlı model.
# ++"HardlyHumans/Facial-expression-detection" → FER-2013 + AffectNet ile eğitilmiş, 8 sınıfı destekleyen gelişmiş ViT modeli.
# -"dima806/facial_emotions_image_detection"  → %90 doğruluk bildiren, FER-2013 benzeri stabil ViT modeli.
# +"prithivMLmods/Facial-Emotion-Detection-SigLIP2" → SigLIP2 tabanlı hızlı model; sınıf sayısı az ama inference hızlı.
# "WongKinYiu/yolov7-w6-emotion"           → Yüz tespiti + duygu analizi aynı anda yapan YOLOv7 tabanlı model.
# "Hosseinali/VisionEmotionNet"            → Derin CNN ile sabit pozlarda yüksek doğruluklu yüz ifadeleri.

###############################################

###############################################
# LOGIN VE İNTERNET KAPAMA
###############################################

login(token=HF_TOKEN)

def disable_internet():
    system = platform.system().lower()
    print(">> İnternet kapatılıyor...")
    try:
        if "windows" in system:
            os.system('netsh interface set interface "Wi-Fi" admin=disabled')
            os.system('netsh interface set interface "Ethernet" admin=disabled')
        elif "linux" in system:
            os.system("nmcli networking off")
        elif "darwin" in system:
            os.system("networksetup -setairportpower Wi-Fi off")
    except Exception as e:
        print(f"İnternet kapatma başarısız: {e}")

###############################################
# MODEL YÜKLEME
###############################################

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
print("Model yüklendi. İnternet kapatılıyor...")
disable_internet()

###############################################
# ANALİZ FONKSİYONU
###############################################

def analyze_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    label = model.config.id2label[pred_idx]
    score = probs[pred_idx].item()
    return label, score

###############################################
# VIDEO / WEBCAM İŞLEME
###############################################

cap = cv2.VideoCapture(0 if use_webcam else video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
results = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, score = analyze_frame(frame)

    timestamp = frame_idx / fps  # saniye cinsinden
    results.append({
        "time_sec": round(timestamp, 2),
        "emotion": label,
        "score": round(score, 4)
    })

    # Ekranda göster
    cv2.putText(frame, f"{label} ({score:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

###############################################
# JSON OLARAK KAYDET
###############################################

with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nAnaliz tamamlandı. Sonuçlar '{output_json}' dosyasına kaydedildi.")
