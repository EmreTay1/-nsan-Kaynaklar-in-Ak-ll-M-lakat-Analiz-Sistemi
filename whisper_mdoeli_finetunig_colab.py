"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Türkçe Whisper Fine-tuning Scripti (MediaSpeech Turkish)
Model: openai/whisper-large-v3
Amaç: MediaSpeech Turkish veri seti (zip halinde) üzerinde ASR fine-tuning yapmak.
"""

# ======================================================
# 1️⃣ GEREKLİ KÜTÜPHANELER
# ======================================================
import os
import zipfile
import torch
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# ======================================================
# 2️⃣ PARAMETRELER
# ======================================================
MODEL_NAME = "openai/whisper-large-v3"
SAVE_DIR = "/content/whisper_turkish_finetuned_MediaSpeech_Turkish"
ZIP_PATH = "/content/TR.zip"  # MediaSpeech zip dosya yolu
EXTRACT_DIR = "/content/MediaSpeech_Turkish"   # Zipten çıkacak klasör
MAX_INPUT_LENGTH = 30.0  # saniye cinsinden maksimum ses uzunluğu

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 3️⃣ ZIP DOSYASINI ÇIKAR
# ======================================================
if not os.path.exists(EXTRACT_DIR):
    print(f"📦 Zip dosyası açılıyor: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"✅ Zip dosyası çıkarıldı: {EXTRACT_DIR}")
else:
    print(f"✅ Zip daha önce çıkarılmış: {EXTRACT_DIR}")

# ======================================================
# 4️⃣ MODEL VE PROCESSOR YÜKLE
# ======================================================
print("🧠 Whisper modeli yükleniyor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)
print(f"✅ Model yüklendi. Kullanılan cihaz: {device}")

# ======================================================
# 5️⃣ DATASET OLUŞTUR (MediaSpeech Turkish)
# ======================================================
print("🎧 Dataset hazırlanıyor...")

audio_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".flac")]
dataset_list = []

for audio_file in audio_files:
    base_name = os.path.splitext(audio_file)[0]
    text_file = base_name + ".txt"
    text_path = os.path.join(EXTRACT_DIR, text_file)
    audio_path = os.path.join(EXTRACT_DIR, audio_file)

    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        dataset_list.append({"audio": {"path": audio_path}, "sentence": transcript})

dataset = Dataset.from_list(dataset_list)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ======================================================
# 6️⃣ ÖN İŞLEME (Preprocessing)
# ======================================================
def prepare_dataset(batch):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )

    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt")

    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels.input_ids[0]
    return batch

print("🔄 Dataset işleniyor (bu işlem biraz zaman alabilir)...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)

# ======================================================
# 7️⃣ METRİKLER
# ======================================================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ======================================================
# 8️⃣ EĞİTİM PARAMETRELERİ
# ======================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/results_whisper_tr",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    save_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    save_steps=200,
    eval_steps=200,
    push_to_hub=False,
)

# ======================================================
# 9️⃣ TRAINER OLUŞTUR VE EĞİTİM
# ======================================================
print("🚀 Fine-tuning başlatılıyor...")

# Dataset’i train/eval olarak ayırıyoruz
train_dataset = dataset.shuffle(seed=42).select(range(min(500, len(dataset))))
eval_dataset = dataset.shuffle(seed=42).select(range(min(100, len(dataset))))

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {
        "input_features": torch.stack([f["input_features"] for f in data]),
        "labels": torch.stack([f["labels"] for f in data]),
    },
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# ======================================================
# 🔟 MODELİ KAYDET
# ======================================================
print("\n💾 Model kaydediliyor...")
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"✅ Model kaydedildi: {SAVE_DIR}")
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Türkçe Whisper Fine-tuning Scripti (OpenSLR108 Turkish)
Model: openai/whisper-large-v3
Amaç: OpenSLR108 Turkish veri seti üzerinde ASR fine-tuning yapmak.
"""
#ne yapıldı  openai/whisper-large-v3 modeline TR.zip data setiyle fine tuning yapılıyor
# ======================================================
# 1️⃣ GEREKLİ KÜTÜPHANELER
# ======================================================
import torch
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# ======================================================
# 2️⃣ PARAMETRELER
# ======================================================
MODEL_NAME = "openai/whisper-large-v3"
SAVE_DIR = "/content/whisper_turkish_finetuned_OpenSLR108"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 3️⃣ MODEL VE PROCESSOR YÜKLE
# ======================================================
print("🧠 Whisper modeli yükleniyor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)
print(f"✅ Model yüklendi. Kullanılan cihaz: {device}")

# ======================================================
# 4️⃣ DATASET YÜKLE (Hugging Face üzerinden OpenSLR108 Turkish)
# ======================================================
print("🎧 Dataset yükleniyor...")
dataset = load_dataset("emre/Open_SLR108_Turkish_10_hours")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ======================================================
# 5️⃣ ÖN İŞLEME (Preprocessing)
# ======================================================
def prepare_dataset(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels.input_ids[0]
    return batch

print("🔄 Dataset işleniyor...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)

# ======================================================
# 6️⃣ METRİKLER
# ======================================================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ======================================================
# 7️⃣ EĞİTİM PARAMETRELERİ
# ======================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/results_whisper_tr",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    save_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    save_steps=200,
    eval_steps=200,
    push_to_hub=False,
)

# ======================================================
# 8️⃣ TRAINER OLUŞTUR VE EĞİTİM
# ======================================================
print("🚀 Fine-tuning başlatılıyor...")

train_dataset = dataset["train"].shuffle(seed=42).select(range(min(500, len(dataset["train"]))))
eval_dataset = dataset["validation"].shuffle(seed=42).select(range(min(100, len(dataset["validation"]))))

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {
        "input_features": torch.stack([f["input_features"] for f in data]),
        "labels": torch.stack([f["labels"] for f in data]),
    },
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# ======================================================
# 9️⃣ MODELİ KAYDET
# ======================================================
print("\n💾 Model kaydediliyor...")
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)