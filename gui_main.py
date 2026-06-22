"""Basit Tkinter GUI: video seç, dil seç, açık rıza onayı, süreci başlat."""

import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk  # Combobox için gerekli
from tkinter import filedialog, messagebox, scrolledtext

# Mevcut modülleriniz
import config
from biometric_pipeline import biyometrik_calistir


class _QueueWriter:
    """stdout/stderr'i GUI log kutusuna yönlendirmek için basit bir writer."""

    def __init__(self, q, fallback_stream):
        self.q = q
        self.fallback_stream = fallback_stream

    def write(self, msg):
        if msg:
            self.q.put(msg)
            try:
                self.fallback_stream.write(msg)
            except Exception:
                pass

    def flush(self):
        try:
            self.fallback_stream.flush()
        except Exception:
            pass


DISCLAIMER = (
    "Bu sistem yapay zeka destekli bir mülakat analiz aracıdır. "
    "Üretilen çıktılar nihai değerlendirme değildir; karar verici insan "
    "kaynakları uzmanlarıdır. Bu araç yalnızca karar destekleyicidir."
)

ACIK_RIZA_METNI = (
    """
KVKK KAPSAMINDA AYDINLATMA VE AÇIK RIZA METNİ

Bu uygulama, işe alım süreçlerinde aday yetkinliklerini değerlendirmek amacıyla Yapay Zeka destekli bir analiz aracı olarak kullanılmaktadır.

Veri İşleme Süreci:
1. Biyometrik Analiz: Mülakat sırasındaki görüntü ve ses verileriniz işlenerek duygu durumu ve stres analizi yapılacaktır.
2. İçerik Analizi: Ses kayıtlarınız metne dönüştürülecek (transkript) ve kişisel bilgileriniz (isim, kurum vb.) [ADAY_ISMI] şeklinde anonimleştirilerek yetkinlik analizi yapılacaktır.

Veri Güvenliği ve İmha Garantisi:
İşlenen ham video ve ses dosyaları, analiz raporu oluşturulduğu an sistemden kalıcı olarak ve güvenli silme yöntemleriyle (Secure Wipe) imha edilecektir. Hiçbir biyometrik veri saklanmayacaktır.

Yasal Haklarınız:
6698 sayılı KVKK'nın 11. maddesi uyarınca, otomatik sistemler vasıtasıyla analiz edilme sonucuna itiraz etme hakkınız saklıdır.

Yukarıdaki süreçleri okudum, biyometrik verilerimin anlık analiz için işlenmesine ve analiz sonrası imha edilmesine AÇIK RIZA gösteriyorum.
    """
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yerel Mülakat Analizi - GUI")
        self.geometry("720x800")  # Dil seçeneği eklendiği için boyutu biraz arttırdım

        self.video_path_var = tk.StringVar(value=config.VIDEO_DOSYASI)
        self.language_var = tk.StringVar(value="Türkçe")  # Varsayılan dil
        self.consent_var = tk.BooleanVar(value=False)
        self.log_queue = queue.Queue()
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

        self._build_ui()
        self._process_log_queue()

    def _build_ui(self):
        # --- BÖLÜM 1: Video Seçimi ---
        tk.Label(self, text="1. Mülakat Videosu Seçin", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=16, pady=(16, 4)
        )

        path_frame = tk.Frame(self)
        path_frame.pack(fill="x", padx=16)
        tk.Entry(path_frame, textvariable=self.video_path_var, width=60).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(path_frame, text="Gözat", command=self._select_file).pack(
            side="left", padx=6
        )

        # --- BÖLÜM 2: Dil Seçimi (YENİ EKLENEN KISIM) ---
        tk.Label(self, text="2. Mülakat Dili", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=16, pady=(16, 4)
        )

        lang_frame = tk.Frame(self)
        lang_frame.pack(fill="x", padx=16)

        # Combobox (Açılır Liste)
        lang_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=["Türkçe", "English"],
            state="readonly",
            width=20
        )
        lang_combo.pack(side="left")

        tk.Label(lang_frame, text="(Whisper modeli ve analiz bu dile göre yapılandırılacaktır)", fg="gray",
                 font=("Arial", 8)).pack(side="left", padx=10)

        # --- BÖLÜM 3: Açık Rıza Metni ---
        tk.Label(self, text="3. Açık Rıza Metni", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=16, pady=(16, 4)
        )

        # Metin ve Scrollbar'ı tutacak bir çerçeve (Frame)
        consent_frame = tk.Frame(self)
        consent_frame.pack(fill="both", expand=False, padx=16)

        # Scrollbar oluştur
        scrollbar = tk.Scrollbar(consent_frame)
        scrollbar.pack(side="right", fill="y")

        # Text alanı oluştur ve scrollbar'a bağla
        consent_text = tk.Text(
            consent_frame,
            height=8,
            wrap="word",
            yscrollcommand=scrollbar.set
        )
        consent_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=consent_text.yview)

        consent_text.insert("1.0", ACIK_RIZA_METNI)
        consent_text.configure(state="disabled")  # Salt okunur yap

        # Onay Kutusu
        tk.Checkbutton(
            self,
            text="Açık rızayı okudum, onaylıyorum.",
            variable=self.consent_var,
        ).pack(anchor="w", padx=16, pady=(8, 12))

        # --- BÖLÜM 4: Yasal Uyarı (Disclaimer) ---
        tk.Label(self, text="Çıktı Hakkında", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=16, pady=(4, 4)
        )
        disclaimer = tk.Text(self, height=4, wrap="word", fg="darkred")
        disclaimer.pack(fill="both", expand=False, padx=16)
        disclaimer.insert("1.0", DISCLAIMER)
        disclaimer.configure(state="disabled")

        # --- BÖLÜM 5: Log Ekranı ---
        tk.Label(self, text="İşlem Logları", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=16, pady=(4, 4)
        )
        self.log_text = scrolledtext.ScrolledText(self, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.log_text.configure(state="disabled")

        # --- BÖLÜM 6: Butonlar ---
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=16, pady=12)

        self.btn_start = tk.Button(
            btn_frame, text="ANALİZİ BAŞLAT", command=self._start_process, width=25, height=2, bg="#dddddd"
        )
        self.btn_start.pack(side="left")

        tk.Button(btn_frame, text="Kapat", command=self.destroy, width=12).pack(
            side="right"
        )

        self.status_var = tk.StringVar(value="Hazır.")
        tk.Label(self, textvariable=self.status_var, fg="blue").pack(
            anchor="w", padx=16, pady=(0, 8)
        )

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Mülakat videosu seç",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if path:
            self.video_path_var.set(path)

    def _start_process(self):
        if not self.consent_var.get():
            messagebox.showwarning("Onay gerekli", "Lütfen açık rıza kutusunu işaretleyin.")
            return

        video_path = self.video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Dosya gerekli", "Lütfen bir video dosyası seçin.")
            return

        self.status_var.set("Analiz başlatıldı, lütfen bekleyin...")
        self._disable_ui()

        self._enable_log_redirect()
        threading.Thread(
            target=self._run_pipeline_safe, args=(video_path,), daemon=True
        ).start()

    def _run_pipeline_safe(self, video_path: str):
        try:
            # Seçilen dili koda çeviriyoruz
            secilen_dil = self.language_var.get()
            dil_kodu = "tr" if secilen_dil == "Türkçe" else "en"

            print(f"--- İşlem Başlıyor ---")
            print(f"Seçilen Video: {video_path}")
            print(f"Seçilen Dil: {secilen_dil} (Kod: {dil_kodu})")

            # Pipeline'ı seçilen dil kodu ile çalıştırıyoruz
            islem_uuid = biyometrik_calistir(dil=dil_kodu, video_path=video_path)

            self.status_var.set(f"Analiz tamamlandı.")

            messagebox.showinfo(
                "Bitti",
                f"Analiz başarıyla tamamlandı.\nIslemUUID: {islem_uuid}\n\n"
                "Rapor oluşturuldu, geçici dosyalar ve orjinal video imha edildi."
            )
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Hata: {exc}")
            messagebox.showerror("Hata", f"Analiz sırasında hata: {exc}")
        finally:
            self._disable_log_redirect()
            self._enable_ui()

    def _disable_ui(self):
        # Arayüz elemanlarını pasif yap (log hariç)
        for child in self.winfo_children():
            try:
                if child != self.log_text:
                    child.configure(state="disabled")
            except Exception:
                pass
        self.update_idletasks()

    def _enable_ui(self):
        # Arayüz elemanlarını aktif yap
        for child in self.winfo_children():
            try:
                child.configure(state="normal")
            except Exception:
                pass
        # Log text varsayılan olarak disabled olmalı (sadece kod yazabilsin diye)
        self.log_text.configure(state="disabled")
        # Disclaimer ve Consent text de disabled kalmalı (read-only)
        # Bu yüzden onları tekrar disable etmemiz gerekebilir ama
        # basitlik adına kullanıcı elle yazamaz zaten.
        self.update_idletasks()

    def _enable_log_redirect(self):
        sys.stdout = _QueueWriter(self.log_queue, self.orig_stdout)
        sys.stderr = _QueueWriter(self.log_queue, self.orig_stderr)

    def _disable_log_redirect(self):
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr

    def _process_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(120, self._process_log_queue)


def run_gui():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run_gui()