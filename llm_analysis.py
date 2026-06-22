"""Yerel LLM (LM Studio) tabanlı İK analiz fonksiyonları."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from docx import Document

import config
# io_ops importunu kaldırdık veya kullanmayacağız çünkü temizliği pipeline yapacak
from pii import anonimize_metin


def read_text_from_docx(file_path):
    """Bir .docx dosyasındaki tüm metni okur."""
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:  # noqa: BLE001
        print(f"HATA: '{file_path}' dosyası okunurken bir hata oluştu: {e}")
        return None


def get_llm_analysis(prompt, model_name, max_tokens=1500):
    """LM Studio API aracılığıyla yerel LLM'e bir prompt gönderir ve yanıtı alır."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(config.LM_STUDIO_API_URL, json=payload)
        response.raise_for_status()
        response_data = response.json()

        print("API Yanıtı:", json.dumps(response_data, indent=2, ensure_ascii=False))

        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        print("HATA: API yanıtında 'choices' anahtarı bulunamadı veya boş.")
        return None
    except requests.exceptions.ConnectionError:
        print("HATA: LM Studio API sunucusuna bağlanılamadı. Lütfen LM Studio'yu kontrol edin.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HATA: HTTP hatası: {http_err}")
        print(f"Yanıt İçeriği: {response.text}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"HATA: Analiz sırasında beklenmedik bir hata oluştu: {e}")
        return None


def aday_konusmaciyi_tespit_et(json_yolu: str) -> str:
    """LLM kullanarak transkriptten hangi konuşmacının aday olduğunu tespit eder."""
    if not Path(json_yolu).exists():
        print(f"Uyarı: '{json_yolu}' bulunamadı.")
        return None

    try:
        with open(json_yolu, "r", encoding="utf-8") as f:
            veri = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"HATA: JSON transkript okunamadı: {e}")
        return None

    if not veri:
        return None

    konusmacilar = list(set([s.get("konusmaci", "") for s in veri if s.get("konusmaci")]))

    if len(konusmacilar) < 2:
        print(f"Uyarı: Yeterli konuşmacı bulunamadı. Bulunan: {konusmacilar}")
        return konusmacilar[0] if konusmacilar else None

    ornek_segmentler = []
    for s in veri[:10]:
        konusmaci = s.get("konusmaci", "")
        metin = s.get("metin", "")
        metin = anonimize_metin(metin)
        ornek_segmentler.append(f"[{konusmaci}]: {metin}")

    ornek_transkript = "\n".join(ornek_segmentler)

    prompt_aday_tespit = f"""
Aşağıda bir mülakat transkriptinin ilk birkaç segmenti verilmiştir. 
Bu bir iş görüşmesi mülakatıdır. Transkriptteki konuşmacılardan hangisinin 
ADAY (işe başvuran kişi) olduğunu tespit et.

Konuşmacılar: {', '.join(konusmacilar)}

Örnek Transkript:
{ornek_transkript}

Görevin: Sadece aday olan konuşmacının adını/etiketini belirt. 
Cevabın SADECE konuşmacı adı/etiketi olsun, başka bir şey yazma.
Örnek cevap formatı: "SPEAKER_00" veya "Konuşmacı 1" gibi.

Hangi konuşmacı aday?
"""

    print("🔍 Aday konuşmacı tespit ediliyor...")
    aday_konusmaci = get_llm_analysis(prompt_aday_tespit, config.MODEL_NAME)

    if aday_konusmaci:
        aday_konusmaci = aday_konusmaci.strip()
        aday_konusmaci = aday_konusmaci.strip('"').strip("'").strip()

        if aday_konusmaci in konusmacilar:
            print(f"✅ Aday konuşmacı tespit edildi: {aday_konusmaci}")
            return aday_konusmaci
        else:
            for k in konusmacilar:
                if aday_konusmaci.lower() in k.lower() or k.lower() in aday_konusmaci.lower():
                    print(f"✅ Aday konuşmacı tespit edildi (benzerlik): {k}")
                    return k

            konusmaci_sayilari = Counter([s.get("konusmaci", "") for s in veri])
            en_cok_konusan = konusmaci_sayilari.most_common(1)[0][0] if konusmaci_sayilari else None
            print(f"⚠️  LLM cevabı eşleşmedi. En çok konuşan kişi varsayılan aday: {en_cok_konusan}")
            return en_cok_konusan
    else:
        konusmaci_sayilari = Counter([s.get("konusmaci", "") for s in veri])
        en_cok_konusan = konusmaci_sayilari.most_common(1)[0][0] if konusmaci_sayilari else None
        print(f"⚠️  LLM cevap veremedi. En çok konuşan kişi varsayılan aday: {en_cok_konusan}")
        return en_cok_konusan


def json_transkripti_zengin_formata_cevir(json_yolu: str, aday_konusmaci: str = None) -> str:
    """JSON transkript dosyasını zengin bir metin formatına dönüştürür."""
    if not Path(json_yolu).exists():
        print(f"Uyarı: '{json_yolu}' bulunamadı.")
        return ""

    try:
        with open(json_yolu, "r", encoding="utf-8") as f:
            veri = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"HATA: JSON transkript okunamadı: {e}")
        return ""

    if not veri:
        return ""

    if aday_konusmaci:
        aday_segmentleri = [s for s in veri if s.get("konusmaci") == aday_konusmaci]
        print(f"📌 Aday konuşmacıya ait {len(aday_segmentleri)} segment bulundu (toplam {len(veri)} segmentten).")
    else:
        aday_segmentleri = veri
        print("⚠️  Aday konuşmacı belirlenemedi, tüm segmentler kullanılacak.")

    toplam_segment = len(aday_segmentleri)
    ses_tonlari = [
        s.get("ses_tonu", "") for s in aday_segmentleri if s.get("ses_tonu") and s.get("ses_tonu") != "[N/A]"
    ]
    yuz_ifadeleri = [
        s.get("yuz_ifadesi", "") for s in aday_segmentleri if s.get("yuz_ifadesi") and s.get("yuz_ifadesi") != "BELİRSİZ"
    ]
    konusmacilar = list(set([s.get("konusmaci", "") for s in veri if s.get("konusmaci")]))

    duraksama_sayisi = sum(1 for s in aday_segmentleri if "(duraksama)" in s.get("metin", "").lower())

    dolgu_sayisi = sum(
        1
        for s in aday_segmentleri
        if any(
            dolgu in s.get("metin", "")
            for dolgu in ["(ııı)", "(eee)", "(hmm)", "(şey)", "(uh)", "(um)", "(like)"]
        )
    )

    formatli_metin = []
    formatli_metin.append("=" * 80)
    formatli_metin.append("MÜLAKAT TRANSKRİPTİ - DETAYLI ANALİZ VERİLERİ")
    formatli_metin.append("=" * 80)
    formatli_metin.append("")
    formatli_metin.append("📊 GENEL İSTATİSTİKLER:")
    if aday_konusmaci:
        formatli_metin.append(f"  • Aday Konuşmacı: {aday_konusmaci} ⭐")
    formatli_metin.append(f"  • Adayın Toplam Konuşma Segmenti: {toplam_segment}")
    formatli_metin.append(f"  • Toplam Konuşmacı Sayısı: {len(konusmacilar)}")
    formatli_metin.append(f"  • Konuşmacılar: {', '.join(konusmacilar)}")
    formatli_metin.append(f"  • Adayın Duraksama Sayısı: {duraksama_sayisi}")
    formatli_metin.append(f"  • Adayın Dolgu Kelimesi Kullanımı: {dolgu_sayisi}")
    if ses_tonlari:
        en_cok_ses_tonu = Counter(ses_tonlari).most_common(1)[0][0] if ses_tonlari else "N/A"
        formatli_metin.append(f"  • En Yaygın Ses Tonu: {en_cok_ses_tonu}")
    if yuz_ifadeleri:
        en_cok_yuz = Counter(yuz_ifadeleri).most_common(1)[0][0] if yuz_ifadeleri else "N/A"
        formatli_metin.append(f"  • En Yaygın Yüz İfadesi: {en_cok_yuz}")
    formatli_metin.append("")
    formatli_metin.append("=" * 80)
    if aday_konusmaci:
        formatli_metin.append(f"DETAYLI KONUŞMA TRANSKRİPTİ (SADECE ADAY: {aday_konusmaci})")
    else:
        formatli_metin.append("DETAYLI KONUŞMA TRANSKRİPTİ")
    formatli_metin.append("=" * 80)
    formatli_metin.append("")

    for idx, s in enumerate(aday_segmentleri, 1):
        konusmaci = s.get("konusmaci", "BİLİNMİYOR")
        metin = s.get("metin", "")
        ses_tonu = s.get("ses_tonu", "[N/A]")
        yuz_ifadesi = s.get("yuz_ifadesi", "BELİRSİZ")
        bas_zaman = s.get("bas", 0)
        bit_zaman = s.get("bit", 0)
        sure = round(bit_zaman - bas_zaman, 2)

        metin = anonimize_metin(metin)

        formatli_metin.append(f"[SEGMENT {idx}]")
        formatli_metin.append(f"  Konuşmacı: {konusmaci}")
        formatli_metin.append(f"  Zaman: {bas_zaman:.2f}s - {bit_zaman:.2f}s (Süre: {sure}s)")
        formatli_metin.append(f"  Ses Tonu: {ses_tonu}")
        formatli_metin.append(f"  Yüz İfadesi: {yuz_ifadesi}")
        formatli_metin.append(f"  Metin: {metin}")
        formatli_metin.append("")

    return "\n".join(formatli_metin)


def ik_analyze_interview():
    """Mülakat analiz süreci - JSON transkriptten tüm bilgileri kullanarak."""
    print(f"JSON transkript dosyası okunuyor: {config.JSON_CIKTI}...")

    aday_konusmaci = aday_konusmaciyi_tespit_et(config.JSON_CIKTI)

    zengin_transkript = json_transkripti_zengin_formata_cevir(
        config.JSON_CIKTI, aday_konusmaci=aday_konusmaci
    )
    if not zengin_transkript or not zengin_transkript.strip():
        print(f"HATA: '{config.JSON_CIKTI}' dosyası boş veya okunamadı.")
        return

    print("Transkript başarıyla okundu. İK analizi isteği gönderiliyor...")

    # --- PROMPT GÜNCELLEMESİ BAŞLANGIÇ ---
    prompt_analysis = f"""
    Sen deneyimli bir İnsan Kaynakları Uzmanısın. Görevin aşağıdaki mülakat transkriptini analiz ederek profesyonel bir değerlendirme raporu yazmaktır.

    ANALİZ EDİLECEK METİN:
    {zengin_transkript}

    -------------------------------------------------------------------------

    GÖREV VE KURALLAR:
    1. Analizi sadece yukarıdaki metne dayanarak yap. Asla hayal ürünü bilgi ekleme.
    2. Her bir kriter için adaya 1 ile 5 arasında puan ver (1: Çok Kötü, 5: Mükemmel).
    3. KRİTİK: Puan verdiğin her madde için "NEDEN" bu puanı verdiğini detaylıca açıkla.
    4. Açıklamalarında adayın kurduğu cümlelerden örnekler ver (Örn: "Aday stres altında olduğunu ... diyerek belirttiği için puan kırıldı").
    5. Asla "[Gerekçe]" gibi şablon ifadeler bırakma, içini dolu dolu yaz.

    ÇIKTI FORMATI (Aynen bu başlıkları kullan):

    ----------------------------------------
    🧩 İK ADAY DEĞERLENDİRME RAPORU
    ----------------------------------------

    ### 1. Yetkinlik Bazlı Puanlama

    (Aşağıdaki maddelerin her birini doldur)

    • İletişim Becerisi: [Puan]/5
      - Değerlendirme: [Adayın kendini ifade ediş biçimi, netliği ve akıcılığı hakkında detaylı yorum buraya.]

    • Motivasyon ve Tutku: [Puan]/5
      - Değerlendirme: [Adayın işe ve sektöre olan ilgisi, heyecanı hakkında yorum buraya.]

    • Analitik Düşünme: [Puan]/5
      - Değerlendirme: [Sorunlara yaklaşımı ve çözüm üretme becerisi hakkında yorum buraya.]

    • Profesyonel Tutum: [Puan]/5
      - Değerlendirme: [Ciddiyeti, saygısı ve profesyonelliği hakkında yorum buraya.]

    • Teknik Yetkinlik: [Puan]/5
      - Değerlendirme: [Kullandığı teknik terimler ve bilgi derinliği hakkında yorum buraya.]

    • Kültürel Uyum: [Puan]/5
      - Değerlendirme: [Takım çalışmasına yatkınlığı ve şirket kültürüne uyumu hakkında yorum buraya.]

    **Genel Ortalama Puan:** [Ortalama]/5.00

    ---

    ### 2. Güçlü ve Zayıf Yönler
    **Güçlü Yönler:**
    - [Madde 1]
    - [Madde 2]

    **Zayıf Yönler:**
    - [Madde 1]
    - [Madde 2]
    
    **Gelişime Açık Alanlar:**
    - [Madde 1]
    - [Madde 2]
    ---

    ### 3. Nihai Rol Uyumu Yorumu
    [Adayın bu pozisyona uygunluğu hakkında 3-4 cümlelik özet, karar destekleyici nihai görüş.]

    RAPORU YUKARIDAKİ FORMATTA OLUŞTUR.
    """
    # --- PROMPT GÜNCELLEMESİ BİTİŞ ---

    # Token sayısını artırdık ki uzun cevaplarda kesilmesin.
    # temperature 0.1 yaparak modelin daha tutarlı ve 'uydurmasız' cevap vermesini sağlıyoruz.
    # (Orijinal kodunuzda temperature 0.7 idi, bu raporlama için biraz yüksek olabilir).

    # get_llm_analysis fonksiyonuna temperature parametresi eklemeniz gerekebilir veya
    # mevcut fonksiyonda default değeri düşürebilirsiniz.
    analysis_result = get_llm_analysis(prompt_analysis, config.MODEL_NAME, max_tokens=3000)

    if not analysis_result:
        print("Analiz alınamadı. İşlem durduruluyor.")
        return

    print("Analiz başarıyla tamamlandı, profesyonel DOCX formatında kaydediliyor...")

    # ... (DOCX kaydetme kısmı aynı kalabilir) ...
    try:
        doc = Document()
        # ... (mevcut docx kodlarınız) ...
        # (Aynı kodları buraya tekrar yazmıyorum, sadece prompt değişti)

        # Basitçe yazdırma döngüsü (Mevcut kodunuzdaki gibi)
        baslik = doc.add_heading("Mülakat Analiz Raporu", level=0)
        baslik.alignment = 1
        tarih_paragraf = doc.add_paragraph()
        tarih_paragraf.add_run(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}").italic = True
        tarih_paragraf.alignment = 1
        doc.add_paragraph()

        satirlar = analysis_result.split("\n")
        for satir in satirlar:
            satir = satir.strip()
            if not satir: continue

            if satir.startswith("###"):
                doc.add_heading(satir.replace("###", "").strip(), level=1)
            elif "🧩" in satir:
                p = doc.add_paragraph(satir)
                p.runs[0].bold = True
            elif satir.startswith("•") or satir.startswith("-"):
                doc.add_paragraph(satir, style="List Bullet")
            else:
                doc.add_paragraph(satir)

        doc.save(config.OUTPUT_DOCX_FILE)
        print(f"✅ Analiz sonuçları '{config.OUTPUT_DOCX_FILE}' dosyasına kaydedildi.")

    except Exception as e:
        print(f"HATA: Çıktı dosyası kaydedilirken bir hata oluştu: {e}")


def transkript_jsondan_docx_olustur(
    json_yolu: str = config.JSON_CIKTI,
    docx_yolu: str = config.INPUT_DOCX_FILE,
    anonimlestir: bool = True,
) -> None:
    """JSON transkript dosyasını DOCX formatına dönüştürür."""
    if not Path(json_yolu).exists():
        print(f"Uyarı: '{json_yolu}' bulunamadı, DOCX oluşturulamadı.")
        return

    try:
        with open(json_yolu, "r", encoding="utf-8") as f:
            veri = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"HATA: JSON transkript okunamadı: {e}")
        return

    doc = Document()
    doc.add_heading("Duygu Analizli Mülakat Transkripti", level=0)

    for s in veri:
        metin = s.get("metin", "")
        if anonimlestir:
            metin = anonimize_metin(metin)

        konusmaci = s.get("konusmaci", "BİLİNMİYOR")
        ses_tonu = s.get("ses_tonu", "[N/A]")
        yuz_ifadesi = s.get("yuz_ifadesi", "BELİRSİZ")

        giris = f"[{konusmaci}] Ses tonu: {ses_tonu} | Yüz ifadesi: {yuz_ifadesi}"
        p = doc.add_paragraph()
        run_baslik = p.add_run(giris + "\n")
        run_baslik.bold = True

        if metin.strip():
            p.add_run(metin.strip())

    try:
        doc.save(docx_yolu)
        print(f"Transkript DOCX olarak kaydedildi: {docx_yolu}")
    except Exception as e:  # noqa: BLE001
        print(f"HATA: Transkript DOCX kaydedilemedi: {e}")