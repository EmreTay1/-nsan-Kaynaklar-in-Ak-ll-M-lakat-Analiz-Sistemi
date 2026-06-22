"""Central configuration and shared constants for the interview analysis pipeline."""

VIDEO_DOSYASI = "video1723838072.mp4"
WAV_DOSYASI = "gecici_16k_mono.wav"

HF_TOKEN = "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"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

WHISPER_MODEL_ID = "openai/whisper-large-v3"
DUYGU_MODEL = "superb/wav2vec2-large-superb-er"

JSON_CIKTI = "mulakat_transkript.json"
TXT_CIKTI = "mulakat_metin.txt"

SES_ESIK_DB = -40
DURAK_MIN_MS = 400

DOLGU_MIN_SN = 0.12
DOLGU_MAX_SN = 0.60
DOLGU_RMS_MIN = 0.005
DOLGU_RMS_MAX = 0.020
DOLGU_MERGE_GAP = 0.25
MAX_DOLGU_PER_SEGMENT = 1
DOLGU_TR = ["(ııı)", "(eee)", "(hmm)", "(şey)"]
DOLGU_EN = ["(uh)", "(um)", "(hmm)", "(like)"]

DUYGU_SOZLUGU = {
    "angry": "ÖFKELİ",
    "disgust": "TİKSİNMİŞ",
    "fear": "KORKMUŞ",
    "happy": "MUTLU",
    "sad": "ÜZGÜN",
    "surprise": "ŞAŞKIN",
    "neutral": "NÖTR",
}

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "mistral-7b-instruct-v0.3.Q4_0"
INPUT_DOCX_FILE = "duygu_analizli_transkript.docx"
OUTPUT_DOCX_FILE = "analiz_sonucu_yeni_sablon.docx"

PII_ANONYMISER_MODEL = (
    "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
)
PII_CONFIDENCE_THRESHOLD = 0.70

# MSSQL bağlantı ayarları (ImhaLoglari tablosu için)
MSSQL_DRIVER = "ODBC Driver 17 for SQL Server"
DB_SERVER = r"localhost\SQLEXPRESS" # Gerekirse 'localhost,1433' formatında port ekleyin
DB_NAME = "InsanKaynaklariDB"
DB_USER = "sa"
DB_PASSWORD = "YourStrong!Passw0rd"  # Güvenli bir şekilde ortam değişkeniyle geçirin

