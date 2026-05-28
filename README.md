# DS-2 Analyst & Dashboard - Salman Pandu Pandiya

Dokumen ini menjabarkan arsitektur kerja Data Science 2 (DS-2) pada proyek ChatKasir. Ruang lingkup peran ini mencakup perumusan pertanyaan bisnis, eksekusi Exploratory Data Analysis (EDA), dan perancangan visualisasi data. Tanggung jawab ini berlanjut hingga peluncuran dashboard Streamlit dan penyusunan laporan teknis akhir proyek.

## Daftar Isi

- [DS-2 Analyst & Dashboard - Salman Pandu Pandiya](#ds-2-analyst--dashboard---salman-pandu-pandiya)
  - [Daftar Isi](#daftar-isi)
  - [Struktur Folder & File](#struktur-folder--file)
  - [Formulasi Pertanyaan Bisnis](#formulasi-pertanyaan-bisnis)
  - [Exploratory Data Analysis (EDA) & Ekstraksi Wawasan](#exploratory-data-analysis-eda--ekstraksi-wawasan)
    - [Kesenjangan Data Eksplisit](#kesenjangan-data-eksplisit)
    - [Dominasi Segmen Harga](#dominasi-segmen-harga)
    - [Format Kuantitas Acak](#format-kuantitas-acak)
  - [Arsitektur Dashboard (Streamlit)](#arsitektur-dashboard-streamlit)
  - [Laporan Teknis & Finalisasi Dokumentasi](#laporan-teknis--finalisasi-dokumentasi)
  - [Panduan Eksekusi Lokal](#panduan-eksekusi-lokal)

## Struktur Folder & File

Sistem analitik ini dibangun menggunakan arsitektur *root-level* (sejajar) untuk meminimalkan latensi pembacaan data pada mesin peladen Streamlit Cloud.

```text
chatkasir-dashboard/
├── ChatKasir_PB_EDA_Vis_Ins.ipynb  # Basis kode penentuan metrik bisnis, pembersihan data lanjutan, dan EDA
├── app.py                          # Kode sumber antarmuka Streamlit (Ringkasan, Analisis, Simulasi)
├── chatkasir_synthetic.csv         # Pangkalan data final 99.998 baris log simulasi pesanan
├── food_utama.csv                  # Pangkalan data final 18.558 manifes ragam menu lokal
├── slang_utama.csv                 # Pangkalan data final 1.231 entri pemetaan kosakata informal
├── requirements.txt                # Konfigurasi dependensi server (pandas, plotly, streamlit)
└── README.md                       # Dokumentasi teknis terpusat DS-2
```
## Formulasi Pertanyaan Bisnis
Penetapan arah analitik dibangun melalui diskusi tim sejak hari pertama proyek. Kerangka kerja SMART dipadukan dengan formula 4W untuk menghasilkan tujuh pertanyaan bisnis terukur. Penggabungan dua metode ini memastikan setiap metrik visual terikat langsung dengan tindakan operasional nyata untuk para pelaku UMKM.

## Exploratory Data Analysis (EDA) & Ekstraksi Wawasan
Proses identifikasi pola data dieksekusi menggunakan pustaka Pandas dan Seaborn.
- Kesenjangan Data Eksplisit
  Sebanyak 56 persen pelanggan tidak mengetikkan nominal harga pesanan secara langsung. Fakta ini mengubah arah pengembangan NLP. Mesin tidak bisa sekadar membaca teks. Model AI harus dirancang agar mampu melakukan pencarian silang ke pangkalan data menu internal toko.
- Dominasi Segmen Harga
  Analisis pada 43.227 baris data harga menunjukkan konsentrasi transaksi menumpuk di bawah nominal 50 ribu rupiah. Pemilik usaha mendapatkan batasan angka yang jelas. Nominal ini dapat dijadikan patokan harga saat merancang paket promosi (bundling) demi mendongkrak volume penjualan.
- Format Kuantitas Acak
  Teks pelanggan didominasi oleh frasa alfabetikal non-numerik saat menuliskan jumlah pesanan. Kondisi lapangan ini mewajibkan pembuatan fungsi pengonversi teks ke angka sebelum deret kata diproses lebih lanjut oleh jaringan saraf.

## Arsitektur Dashboard (Streamlit)
Aplikasi antarmuka ini dirakit dengan Streamlit dan dapat diakses publik melalui Streamlit Cloud. Sistem terbagi menjadi tiga modul fungsional:
  1. Ringkasan Data: Menampilkan struktur pangkalan data yang menjadi bahan bakar sistem. Pengguna dapat melihat sampel langsung dari 99.998 baris data sintetis, 18.558 nama produk, dan 1.231 kosakata informal.
  2. Analisis Bisnis: Menyajikan grafik Plotly untuk menjawab tujuh pertanyaan bisnis. Panel filter dinamis di sisi kiri layar memungkinkan pemotongan data berdasarkan rentang harga atau pola obrolan. Visualisasi akan merespons perubahan filter secara seketika.
  3. Simulasi ChatKasir: Menyediakan ruang uji algoritma ekstraksi teks. Obrolan kasir dinormalisasi menggunakan logika berbasis aturan (rule-based). Modul ini disiapkan untuk menerima integrasi endpoint API secara langsung dari model NLP utama.

## Panduan Eksekusi Lokal
Jalankan perintah ini di dalam terminal untuk menyalakan server analitik pada mesin Anda:
  1. Tarik kode sumber:
     
     git clone [https://github.com/username-anda/nama-repo.git](https://github.com/username-anda/nama-repo.git)
  2. Pasang pustaka pendukung:
     
     pip install -r requirements.txt
  3. Bangunkan antarmuka web:
     
     streamlit run app.py
