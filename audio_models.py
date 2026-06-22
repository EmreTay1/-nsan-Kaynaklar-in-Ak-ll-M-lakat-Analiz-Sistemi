"""Model yüklemeleri (ses tonu ve Whisper transkripsiyon)."""

import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

import config

print("Ses tonu modeli yükleniyor...")
try:
    extractor = AutoFeatureExtractor.from_pretrained(config.DUYGU_MODEL)
    model_emotion = AutoModelForAudioClassification.from_pretrained(config.DUYGU_MODEL)
    model_emotion.eval()
except Exception as e:  # noqa: BLE001
    print(f"Duygu modeli yüklemede hata: {e}")
    extractor = None
    model_emotion = None

print(f"Whisper modeli yükleniyor: {config.WHISPER_MODEL_ID}...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.WHISPER_MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model_whisper.to(device)
    processor_whisper = AutoProcessor.from_pretrained(config.WHISPER_MODEL_ID)
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
except Exception as e:  # noqa: BLE001
    print(f"Whisper modeli yüklenemedi: {e}")
    transcriber = None

