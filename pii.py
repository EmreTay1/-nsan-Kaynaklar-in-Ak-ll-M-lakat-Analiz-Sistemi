"""PII anonimleştirme yardımcıları."""

import re
from typing import Optional

from transformers import pipeline

import config

try:
    # Opsiyonel: Gerçek PII anonimleştirme için Microsoft Presidio
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _PRESIDIO_AVAILABLE = True
    _presidio_analyzer = AnalyzerEngine()
    _presidio_anonymizer = AnonymizerEngine()
except Exception:  # noqa: BLE001
    _PRESIDIO_AVAILABLE = False

_pii_anonymiser_pipeline: Optional[object] = None

try:
    print("Multilingual PII anonimleştirme modeli yükleniyor...")
    _pii_anonymiser_pipeline = pipeline(
        "token-classification",
        model=config.PII_ANONYMISER_MODEL,
        aggregation_strategy="simple",
    )
    print(
        f"✅ PII anonimleştirme modeli başarıyla yüklendi. "
        f"(Eşik değeri: {config.PII_CONFIDENCE_THRESHOLD})"
    )
except Exception as e:  # noqa: BLE001
    print(f"⚠️  PII anonimleştirme modeli yüklenemedi: {e}")
    _pii_anonymiser_pipeline = None


def anonimize_metin(metin: str) -> str:
    """
    Mülakat metnini PII açısından anonimleştirir (pseudonymization).
    İsim, soyisim, telefon, e-posta ve kurum bilgilerini maskeleme yapar.

    Önce Hugging Face Türkçe PII detection modeli kullanılır,
    ardından Presidio (varsa) ve son olarak regex filtreleri ile ek maskeleme yapılır.
    """
    if not metin:
        return metin

    sonuc = metin

    # 1) ÖNCE: Hugging Face Multilingual PII Anonymiser Modeli (en güçlü yöntem)
    if _pii_anonymiser_pipeline:
        try:
            entities = _pii_anonymiser_pipeline(sonuc)
            if entities:
                entities_sorted = sorted(
                    entities, key=lambda x: x.get("end", 0), reverse=True
                )

                for entity in entities_sorted:
                    score = entity.get("score", 0.0)
                    if score < config.PII_CONFIDENCE_THRESHOLD:
                        continue

                    entity_group = entity.get("entity_group", "").upper()
                    word = entity.get("word", "")
                    start = entity.get("start", 0)
                    end = entity.get("end", 0)

                    if not word or start >= end:
                        continue

                    if entity_group in ["PER", "PERSON", "PERSON_NAME", "NAME"]:
                        replacement = "[ADAY_ISMI]"
                    elif entity_group in ["ORG", "ORGANIZATION", "ORG_NAME", "COMPANY"]:
                        replacement = "[KURUM]"
                    elif entity_group in ["PHONE", "PHONE_NUMBER", "PHONE_NUM", "TEL"]:
                        replacement = "[TELEFON]"
                    elif entity_group in ["EMAIL", "EMAIL_ADDRESS", "EMAIL_ADDR"]:
                        replacement = "[EPOSTA]"
                    elif entity_group in ["LOC", "LOCATION", "ADDRESS"]:
                        replacement = "[KONUM]"
                    else:
                        replacement = "[KISISEL_VERI]"

                    sonuc = sonuc[:start] + replacement + sonuc[end:]
        except Exception:  # noqa: BLE001
            pass

    # 2) Presidio ile ek anonimleştirme (varsa)
    if _PRESIDIO_AVAILABLE:
        try:
            analyzer_results = _presidio_analyzer.analyze(
                text=sonuc,
                language="en",
                entities=["PERSON", "PHONE_NUMBER", "ORGANIZATION", "EMAIL_ADDRESS"],
            )
            anonymized = _presidio_anonymizer.anonymize(
                text=sonuc,
                analyzer_results=analyzer_results,
                anonymizers_config={
                    "PERSON": {"type": "replace", "new_value": "[ADAY_ISMI]"},
                    "PHONE_NUMBER": {"type": "replace", "new_value": "[TELEFON]"},
                    "ORGANIZATION": {"type": "replace", "new_value": "[KURUM]"},
                    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EPOSTA]"},
                },
            )
            sonuc = anonymized.text
        except Exception:  # noqa: BLE001
            pass

    # 3) Ek güvenlik için regex maskeleme (Türkçe isimler ve diğer kalıplar)
    sonuc = re.sub(r"\b\d{10,}\b", "[TELEFON]", sonuc)
    sonuc = re.sub(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}", "[TELEFON]", sonuc)
    sonuc = re.sub(r"\+\d{1,3}[-.\s]?\d{10,}", "[TELEFON]", sonuc)

    sonuc = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EPOSTA]", sonuc
    )

    sonuc = re.sub(
        r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\b",
        r"[ADAY_ISMI]",
        sonuc,
    )
    sonuc = re.sub(
        r"\b(Ben|Benim|Bana|Benden)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,})\b",
        r"\1 [ADAY_ISMI]",
        sonuc,
    )

    return sonuc

