#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Türkçe / İngilizce Ses Tonu Duygu Tanıma Fine-tuning Scripti
Model: superb/wav2vec2-large-superb-er
Amaç: 4 duygulu dataset (angry, disgust, fear, happy, neutral, pleasant_surprise, sad)
üzerinde fine-tuning yapmak.

wav2vec2 tabanlı duygu tanıma modeli (superb/wav2vec2-large-superb-er) için fine tuning
"""

# ======================================================
# 1️⃣ GEREKLİ KÜTÜPHANELER
# ======================================================
import os
import random
import zipfile
import librosa
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch # Import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import EvalPrediction # Import EvalPrediction
import numpy as np # Import numpy

# ======================================================
# 2️⃣ ZIP DOSYASINI AÇ (COLAB İÇİN)
# ======================================================
# Eğer dataset.zip dosası Colab'e yüklüyse:
ZIP_PATH = "/content/Sound Source.zip"    # dataset.zip dosasının yolu
DATA_DIR = "/content/Sound Source/Sound Source"         # açılacağı klasörün içindeki gerçek data klasörü

if os.path.exists(ZIP_PATH):
    print("📦 dataset.zip bulundu, çıkarılıyor...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("/content/Sound Source") # ZIP'i /content/dataset içine çıkar
    print("✅ Dataset başarıyla çıkarıldı:", "/content/dataset")
else:
    print("⚠️ Uyarı: dataset.zip bulunamadı! Lütfen Colab’e yükleyin.")

# ======================================================
# 3️⃣ PARAMETRELER
# ======================================================
TEST_SPLIT_PERCENTAGE = 0.20 # %20 test için
TRAIN_SPLIT_PERCENTAGE = 0.80 # %80 eğitim için
SAMPLING_RATE = 16000
MODEL_NAME = "superb/wav2vec2-large-superb-er"
SAVE_DIR = "/content/emre2_superverb_egitimli"

# ======================================================
# 4️⃣ VERİLERİ HAZIRLA (TRAIN / TEST)
# ======================================================
train_files, train_labels = [], []
test_files, test_labels = [], []

print("\n📂 Veri kümeleri hazırlanıyor...\n")

# DATA_DIR artık /content/dataset/dataset olarak ayarlandı
if not os.path.exists(DATA_DIR):
    print(f"❌ Hata: Beklenen veri klasörü bulunamadı: {DATA_DIR}")
else:
    for emotion in os.listdir(DATA_DIR):
        emotion_dir = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.endswith(".wav")]
        random.shuffle(files)

        # Calculate split based on percentage
        test_count = int(len(files) * TEST_SPLIT_PERCENTAGE)
        train_count = len(files) - test_count # The rest goes to training

        test_files.extend(files[:test_count])
        test_labels.extend([emotion] * test_count)

        train_files.extend(files[test_count:])
        train_labels.extend([emotion] * (len(files) - test_count))


print(f"✅ Train örnek sayısı: {len(train_files)}")
print(f"✅ Test örnek sayısı: {len(test_files)}\n")

if len(train_files) == 0 or len(test_files) == 0:
    print("❌ Hata: Eğitim veya test için hiç örnek bulunamadı. Lütfen veri klasörünün ve içeriğinin doğru olduğundan emin olun.")
    # Scriptin devam etmesini engellemek için bir hata yükseltilebilir veya çıkılabilir
    raise ValueError("Veri setinde yeterli örnek bulunamadı.")


train_dataset = Dataset.from_dict({"path": train_files, "label": train_labels})
test_dataset = Dataset.from_dict({"path": test_files, "label": test_labels})

# ======================================================
# 5️⃣ LABEL ENCODER
# ======================================================
le = LabelEncoder()
le.fit(list(train_dataset["label"]) + list(test_dataset["label"]))

def encode_label(batch):
    batch["label"] = le.transform([batch["label"]])[0]
    return batch

print("🏷️ Etiketler işleniyor...")
train_dataset = train_dataset.map(encode_label)
test_dataset = test_dataset.map(encode_label)
print("✅ Etiketler işlendi.\n")

# ======================================================
# 6️⃣ FEATURE EXTRACTOR
# ======================================================
print("🎧 Özellik çıkarıcı yükleniyor...")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # Hata kontrolü ekleyelim
    if not os.path.exists(batch["path"]):
        print(f"⚠️ Uyarı: Dosya bulunamadı, atlanıyor: {batch['path']}")
        return None # Bu örneği atla

    try:
        audio, sr = librosa.load(batch["path"], sr=SAMPLING_RATE)
        inputs = extractor(audio, sampling_rate=SAMPLING_RATE)
        batch["input_values"] = inputs["input_values"][0]
        return batch
    except Exception as e:
        print(f"❌ Hata işlenirken {batch['path']}: {e}")
        return None # Hata durumında bu örneği atla


print("🎛️ Sesler işleniyor... (birkaç dakika sürebilir)\n")
# map işleminden sonra None döndüren örnekleri filtreleyelim
train_dataset = train_dataset.map(preprocess, remove_columns=["path"]).filter(lambda x: x is not None)
test_dataset = test_dataset.map(preprocess, remove_columns=["path"]).filter(lambda x: x is not None)


# ======================================================
# 7️⃣ MODELİ YÜKLE
# ======================================================
print("🧠 Model yükleniyor...")
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True  # çıkış katmanı uyumsuzluğu düzeltme
)

# Explicitly move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# ======================================================
# 8️⃣ EĞİTİM PARAMETRELERİ
# ======================================================
training_args = TrainingArguments(
    output_dir="/content/results",
    eval_strategy="steps", # Change to steps
    save_strategy="steps",        # Change to steps
    save_steps=500,               # Save every 500 steps
    eval_steps=500,               # Evaluate every 500 steps
    learning_rate=1e-5,
    per_device_train_batch_size=2, # Reduced batch size
    per_device_eval_batch_size=2, # Reduced eval batch size
    num_train_epochs=5,  # Colab için makul süre
    logging_dir="/content/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy", # En iyi modeli seçmek için metrik
    greater_is_better=True, # Daha yüksek doğruluk daha iyidir
    gradient_accumulation_steps=2, # Added gradient accumulation
    gradient_checkpointing=True # Added gradient checkpointing
)

# ======================================================
# 9️⃣ METRİKLER
# ======================================================
def compute_metrics(eval_pred):
    """
    Güçlü bir compute_metrics implementasyonu:
    - eval_pred: transformers.trainer_utils.EvalPrediction veya (predictions, labels) tuple
    - predictions, labels farklı tiplerde (torch.Tensor, np.ndarray, tuple) olabilir.
    - Eğer predictions bir tuple ise (ör: (logits, ...)) ilk eleman alınır.
    """
    # 1) EvalPrediction nesnesi mi yoksa tuple mı?
    if isinstance(eval_pred, EvalPrediction):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    elif isinstance(eval_pred, tuple) and len(eval_pred) >= 2:
        predictions, labels = eval_pred[0], eval_pred[1]
    else:
        raise TypeError("compute_metrics: eval_pred beklenmeyen türde. Beklenen EvalPrediction veya (preds, labels) tuple.")

    # 2) predictions bazen (logits, ...) şeklinde bir tuple olabilir -> ilk elemanı al
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]

    # 3) predictions'i numpy array'e çevir ve argmax uygula
    if isinstance(predictions, torch.Tensor):
        preds = predictions.argmax(dim=-1).cpu().numpy()
    elif isinstance(predictions, np.ndarray):
        preds = predictions.argmax(axis=-1)
    else:
        raise TypeError(f"compute_metrics: predictions için beklenmeyen tip: {type(predictions)}")

    # 4) labels'ı numpy array'e çevir (bazı durumlarda torch tensor gelebilir)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # 5) metrikleri hesapla ve yazdır
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    print(f"📊 Doğruluk: {acc:.4f} | F1 Skoru: {f1:.4f}")

    return {"accuracy": acc, "f1": f1}


# ======================================================
# 🔟 TRAINER VE EĞİTİM
# ======================================================
print("🚀 Fine-tuning başlatılıyor...\n")

trainer = Trainer(
     model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=extractor,
    compute_metrics=compute_metrics
)

trainer.train()

# ======================================================
# 1️⃣1️⃣ MODELİ KAYDET
# ======================================================
model.save_pretrained(SAVE_DIR)
extractor.save_pretrained(SAVE_DIR)

print("\n✅ Eğitim tamamlandı!")
print(f"💾 Model kaydedildi: {SAVE_DIR}")
print("🎯 Etiket sırası:", list(le.classes_))

# ======================================================
# 1️⃣2️⃣ MODELİ ZIPLE VE İNDİR (İSTEĞE BAĞDAŞ)
# ======================================================
print("\n📦 Model zipleniyor...")
os.system(f"zip -r {SAVE_DIR}.zip {SAVE_DIR}")
print(f"✅ Model arşivlendi: {SAVE_DIR}.zip")

from google.colab import files
files.download(f"{SAVE_DIR}.zip")
print("⬇️ Model indirilmeye hazır!")