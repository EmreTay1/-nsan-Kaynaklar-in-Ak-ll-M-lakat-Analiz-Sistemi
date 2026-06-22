"""MSSQL ImhaLoglari tablosuna log yazmak için yardımcılar."""

from datetime import datetime
from typing import Optional
import pyodbc
import config

def _get_connection() -> pyodbc.Connection:
    # Windows Authentication (Şifresiz Bağlantı) kullanıyoruz
    conn_str = (
        f"DRIVER={{{config.MSSQL_DRIVER}}};"
        f"SERVER={config.DB_SERVER};"  # config.py'de localhost\SQLEXPRESS olduğundan emin olun
        f"DATABASE={config.DB_NAME};"
        "Trusted_Connection=yes;"      # Şifre yerine Windows yetkisi kullanılır
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def log_imha_kaydi(
    islem_uuid: str,
    orjinal_video_durumu: Optional[str],
    gecici_dosya_durumu: Optional[str],
    transkript_durumu: Optional[str],
    genel_durum: str,
    tarih: Optional[datetime] = None,
) -> bool:
    """
    ImhaLoglari tablosuna kayıt ekler.
    Bağlantı hatası olursa programı durdurmaz, sadece uyarı verir.
    """
    if not islem_uuid:
        print(" Uyarı: islem_uuid boş, loglama yapılmadı.")
        return False

    tarih = tarih or datetime.utcnow()

    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ImhaLoglari
                    (IslemUUID, Tarih, OrjinalVideoDurumu, GeciciDosyaDurumu, TranskriptDurumu, GenelDurum)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    islem_uuid,
                    tarih,
                    orjinal_video_durumu,
                    gecici_dosya_durumu,
                    transkript_durumu,
                    genel_durum,
                )
                conn.commit()
        print(" ImhaLoglari veritabanı kaydı oluşturuldu.")
        return True

    except pyodbc.Error as db_err:
        # Hata olsa bile programın çökmemesi için hatayı yakalıyoruz
        print(f" Veritabanı Bağlantı Hatası (Loglama atlanıyor): {db_err.args[1] if len(db_err.args) > 1 else db_err}")
        return False
    except Exception as exc:
        print(f" Beklenmeyen veritabanı hatası (Loglama atlanıyor): {exc}")
        return False