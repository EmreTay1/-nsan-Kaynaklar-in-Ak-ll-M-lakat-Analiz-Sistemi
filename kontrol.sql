USE InsanKaynaklariDB;
GO

SELECT TOP 100
    LogID,
    IslemUUID AS [Oturum Kodu],          -- Word dosyasýndaki kodla eþleþecek
    Tarih AS [Ýmha Zamaný],
    OrjinalVideoDurumu AS [Video],       -- PDF Madde 3.3 (Secure Wipe Kanýtý)
    GeciciDosyaDurumu AS [Ses/Temp],
    TranskriptDurumu AS [Transkript],
    GenelDurum
FROM ImhaLoglari
ORDER BY Tarih DESC;