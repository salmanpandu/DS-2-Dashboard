# ==============================================================================
# app.py - ChatKasir Dashboard
# Streamlit: Ringkasan | Analisis | Simulasi
# Berdasarkan notebook: ChatKasir_PB_EDA_Vis_Ins.ipynb
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter

# WAJIB: baris pertama setelah import
st.set_page_config(
    page_title="ChatKasir Analytics",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Konstanta warna brand
C = {
    "biru"  : "#2E86AB",
    "merah" : "#E84855",
    "hijau" : "#3BB273",
    "kuning": "#F7B731",
    "abu"   : "#95A5A6",
}

HARGA_MIN_VALID = 5_000
HARGA_MAX_VALID = 500_000

PETA_KATEGORI = {
    "nasi": "Nasi", "mie": "Mie & Pasta", "spaghetti": "Mie & Pasta",
    "pasta": "Mie & Pasta", "ayam": "Ayam", "chicken": "Ayam",
    "ikan": "Seafood", "udang": "Seafood", "kepiting": "Seafood",
    "cumi": "Seafood", "es": "Minuman Dingin", "ice": "Minuman Dingin",
    "iced": "Minuman Dingin", "kopi": "Kopi", "coffee": "Kopi",
    "teh": "Teh", "juice": "Jus & Smoothie", "jus": "Jus & Smoothie",
    "smoothie": "Jus & Smoothie", "roti": "Roti & Bakery",
    "bolu": "Roti & Bakery", "cake": "Roti & Bakery",
    "donat": "Roti & Bakery", "bakso": "Bakso & Soto",
    "soto": "Bakso & Soto", "sop": "Bakso & Soto",
    "sup": "Bakso & Soto", "pisang": "Buah & Dessert",
    "strawberry": "Buah & Dessert", "mangga": "Buah & Dessert",
    "seblak": "Jajanan", "cireng": "Jajanan", "batagor": "Jajanan",
    "siomay": "Jajanan", "beef": "Daging Sapi", "sapi": "Daging Sapi",
    "steak": "Daging Sapi", "kacang": "Camilan", "keripik": "Camilan",
    "chips": "Camilan", "hot": "Minuman Panas", "choco": "Minuman Panas",
    "chocolate": "Minuman Panas", "martabak": "Jajanan",
    "tahu": "Jajanan", "sosis": "Camilan",
}

URUTAN_SEGMEN = [
    "Murah (< 20rb)",
    "Sedang (20rb-50rb)",
    "Agak Mahal (50rb-100rb)",
    "Mahal (>= 100rb)",
]

# ==============================================================================
# BAGIAN 1 - FUNGSI LOAD & CLEANING DATA (semua di-cache)
# ==============================================================================

@st.cache_data
def muat_sintetis():
    """Muat, bersihkan, dan tambahkan fitur turunan ke data sintetis."""
    df = pd.read_csv("chatkasir_synthetic.csv")

    # Cleaning: hapus duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    # Fitur turunan
    df["harga_eksplisit"] = df["price_satuan"] != -1
    df["harga_valid"]     = (
        (df["price_satuan"] >= HARGA_MIN_VALID) &
        (df["price_satuan"] <= HARGA_MAX_VALID)
    )
    df["harga_outlier"]   = df["harga_eksplisit"] & (df["harga_valid"] == False)
    df["multi_produk"]    = df["product"].str.contains(" & ", na=False)
    df["qty_numerik"]     = pd.to_numeric(
        df["quantity"], errors="coerce"
    ).notna()

    def _segmen(row):
        if not row["harga_valid"]:
            return "Tidak Disebutkan / Outlier"
        h = row["price_satuan"]
        if h < 20_000:    return "Murah (< 20rb)"
        elif h < 50_000:  return "Sedang (20rb-50rb)"
        elif h < 100_000: return "Agak Mahal (50rb-100rb)"
        else:             return "Mahal (>= 100rb)"

    df["segmen_harga"] = df.apply(_segmen, axis=1)
    return df


@st.cache_data
def muat_makanan():
    """Muat dan bersihkan dataset makanan, tambahkan kategori."""
    df = pd.read_csv("food_utama.csv")
    df["name"]        = df["name"].str.strip().str.lower()
    df["name_clean"]  = df["name"]
    df["kategori"]    = df["name_clean"].apply(
        lambda n: PETA_KATEGORI.get(n.split()[0], "Lainnya")
    )
    df["panjang_nama"] = df["name_clean"].apply(lambda n: len(n.split()))
    return df


@st.cache_data
def muat_slang():
    """Muat dan bersihkan dataset slang."""
    df = pd.read_csv("slang_utama.csv")
    df["slang"]  = df["slang"].str.strip().str.lower()
    df["formal"] = df["formal"].str.strip().str.lower()
    df["panjang_slang"]   = df["slang"].str.len()
    df["panjang_formal"]  = df["formal"].str.len()
    df["selisih_panjang"] = df["panjang_formal"] - df["panjang_slang"]
    df["tipe"] = df["selisih_panjang"].apply(
        lambda s: "Disingkat" if s > 0 else ("Sama" if s == 0 else "Diperpanjang")
    )
    return df


@st.cache_data
def hitung_slang_aktif(n_sample: int = 10_000):
    """Hitung frekuensi kata slang dari sampel teks percakapan."""
    df_synth = pd.read_csv("chatkasir_synthetic.csv")
    df_slang = muat_slang()
    set_slang = set(df_slang["slang"])

    sample    = df_synth["input_text"].sample(n_sample, random_state=42)
    semua_kata = []
    for teks in sample:
        bagian_customer = teks.split("[SEP]")[0].lower()
        semua_kata.extend(re.findall(r"\b\w+\b", bagian_customer))

    counter = Counter(k for k in semua_kata if k in set_slang)
    return counter, df_slang


@st.cache_data
def terapkan_filter(harga_min, harga_max, pola_tuple, segmen_tuple):
    """Kembalikan subset data sintetis sesuai filter sidebar."""
    df = muat_sintetis()

    mask_pola = df["pattern"].isin(pola_tuple)
    mask_harga = (
        (df["harga_valid"] == False) |
        (
            (df["price_satuan"] >= harga_min) &
            (df["price_satuan"] <= harga_max)
        )
    )
    mask_segmen = (
        (df["harga_valid"] == False) |
        df["segmen_harga"].isin(segmen_tuple)
    )
    return df[mask_pola & mask_harga & mask_segmen].copy()


# ==============================================================================
# BAGIAN 2 - SIDEBAR: NAVIGASI + FILTER
# ==============================================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#2E86AB; margin-bottom:0'>🧾 ChatKasir</h2>"
        "<p style='color:#666; font-size:12px; margin-top:4px'>"
        "Analytics Dashboard . CC26-PSU065</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    halaman = st.radio(
        "📂 Pilih Halaman",
        options=[
            "🏠 Ringkasan Data",
            "📊 Analisis Bisnis",
            "💬 Simulasi ChatKasir",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if halaman == "📊 Analisis Bisnis":
        st.markdown("### 🎛️ Filter Data")
        st.caption("Filter terhubung ke semua grafik secara real-time")

        harga_min, harga_max = st.slider(
            "Rentang Harga Satuan (Rp)",
            min_value=5_000,
            max_value=500_000,
            value=(5_000, 500_000),
            step=5_000,
            format="Rp %d",
        )

        pola_pilihan = st.multiselect(
            "Pola Percakapan",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Pola 1 - Standar",
                2: "Pola 2 - Informal",
                3: "Pola 3 - Multi Produk Teks",
                4: "Pola 4 - Multi Produk Kompleks",
            }[x],
        )
        if not pola_pilihan:
            st.warning("Pilih minimal 1 pola.")
            pola_pilihan = [1, 2, 3, 4]

        segmen_pilihan = st.multiselect(
            "Segmen Harga",
            options=URUTAN_SEGMEN,
            default=URUTAN_SEGMEN,
        )
        if not segmen_pilihan:
            st.warning("Pilih minimal 1 segmen.")
            segmen_pilihan = URUTAN_SEGMEN

    else:
        harga_min      = 5_000
        harga_max      = 500_000
        pola_pilihan   = [1, 2, 3, 4]
        segmen_pilihan = URUTAN_SEGMEN

    st.markdown("---")
    st.caption("Coding Camp 2026 . DBS Foundation")


df_filtered = terapkan_filter(
    harga_min,
    harga_max,
    tuple(sorted(pola_pilihan)),
    tuple(sorted(segmen_pilihan)),
)


# ==============================================================================
# BAGIAN 3 - HALAMAN 1: RINGKASAN DATA
# ==============================================================================

if halaman == "🏠 Ringkasan Data":

    st.title("🧾 ChatKasir - Ringkasan Dataset")
    st.caption(
        "Ikhtisar tiga dataset yang digunakan sistem ChatKasir: "
        "data sintetis percakapan, kamus slang, dan nama produk makanan Indonesia."
    )

    df_synth = muat_sintetis()
    df_food  = muat_makanan()
    df_slang = muat_slang()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📦 Data Percakapan Sintetis", f"{len(df_synth):,}")
    k2.metric("🍜 Nama Produk Makanan", f"{len(df_food):,}")
    k3.metric("💬 Entri Kamus Slang", f"{len(df_slang):,}")
    k4.metric("🏷️ Kategori Masakan", f"{df_food['kategori'].nunique()}")

    st.markdown("---")

    st.subheader("📋 Formulasi Pertanyaan Bisnis")
    st.markdown(
        "Pertanyaan bisnis merupakan fondasi analitik dasar yang berfungsi sebagai kompas operasional. "
        "Rumusan ini mendefinisikan masalah riil di lapangan, mengarahkan proses pengumpulan bukti, "
        "dan mencegah pencarian pola data acak yang tidak memiliki nilai guna bagi efisiensi UMKM. "
        "Penetapan arah yang jelas ini mendasari keputusan kami mengadopsi metode SMART + 4W "
        "dalam menyusun target evaluasi di bawah ini."
    )
    
    with st.expander("ℹ️ Mengapa Menggunakan Metode SMART + 4W?"):
        st.markdown(
            "Kerangka Kerja SMART menjamin pertanyaan bersifat Spesifik, Measurable (terukur), "
            "Action-oriented (orientasi aksi), Relevan, dan Time-bound (batasan waktu). "
            "Kombinasi dengan formula 4W (What, Why, Where, When) mempertajam visualisasi data "
            "sehingga setiap grafik mampu memicu satu tindakan koreksi operasional yang nyata."
        )

    pb_data = pd.DataFrame([
        {"Kode": "PB-1", "4W": "What", "Pertanyaan": "Apa 20 produk yang paling sering dipesan pelanggan?", "Aksi untuk UMKM": "Prioritaskan stok & promo produk hero"},
        {"Kode": "PB-2", "4W": "What", "Pertanyaan": "Berapa rata-rata & median harga satuan produk eksplisit?", "Aksi untuk UMKM": "Tetapkan benchmark HPP berdasarkan median"},
        {"Kode": "PB-3", "4W": "What", "Pertanyaan": "Bagaimana distribusi segmentasi harga (4 kelompok)?", "Aksi untuk UMKM": "Rancang strategi bundling sesuai segmen dominan"},
        {"Kode": "PB-4", "4W": "Why",  "Pertanyaan": "Mengapa hanya ~43% pesanan menyebut harga eksplisit?", "Aksi untuk UMKM": "Bangun fallback harga dari database"},
        {"Kode": "PB-5", "4W": "What", "Pertanyaan": "Apa kata slang paling aktif dalam percakapan pelanggan?", "Aksi untuk UMKM": "Prioritaskan 30 slang teratas di pipeline AI"},
        {"Kode": "PB-6", "4W": "What", "Pertanyaan": "Bagaimana distribusi tipe kuantitas (angka vs teks)?", "Aksi untuk UMKM": "Bangun konverter teks-angka sebelum masuk model"},
        {"Kode": "PB-7", "4W": "What", "Pertanyaan": "Kategori masakan apa yang mendominasi database produk?", "Aksi untuk UMKM": "Perluas dataset kategori yang under-represented"},
    ])
    st.dataframe(pb_data, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Preview Dataset")
    tab1, tab2, tab3 = st.tabs(["📊 Data Sintetis", "🍜 Data Makanan", "💬 Data Slang"])
    with tab1:
        st.dataframe(df_synth.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df_food.head(20), use_container_width=True)
    with tab3:
        st.dataframe(df_slang.head(20), use_container_width=True)


# ==============================================================================
# BAGIAN 4 - HALAMAN 2: ANALISIS BISNIS
# ==============================================================================

elif halaman == "📊 Analisis Bisnis":
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("📊 Analisis Bisnis - ChatKasir")
    st.caption(f"Menampilkan {len(df_filtered):,} dari 99.998 baris")

    df_hv = df_filtered[df_filtered["harga_valid"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Total Pesanan (Filtered)", f"{len(df_filtered):,}")
    k2.metric("💰 Rata-rata Harga Satuan", f"Rp {df_hv['price_satuan'].mean():,.0f}" if len(df_hv) else "N/A")
    k3.metric("📍 Median Harga", f"Rp {df_hv['price_satuan'].median():,.0f}" if len(df_hv) else "N/A")
    k4.metric("❓ Tanpa Harga Eksplisit", f"{(df_filtered['harga_eksplisit'] == False).sum():,}", delta=f"{(df_filtered['harga_eksplisit'] == False).mean()*100:.1f}%")

    st.markdown("---")

    if not df_filtered.empty:
        col_row1_left, col_row1_right = st.columns(2, gap="medium")
        
        with col_row1_left:
            st.markdown("### 📌 PB-1: Top 20 Produk Terlaris")
            top_produk = df_filtered["product"].value_counts().head(20).reset_index()
            top_produk.columns = ["Produk", "Jumlah Pesanan"]
            fig1 = px.bar(top_produk, x="Jumlah Pesanan", y="Produk", orientation="h", color_discrete_sequence=[C["biru"]])
            fig1.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=380)
            st.plotly_chart(fig1, use_container_width=True)
            st.info("💡 **Explanatory Insight:** Evaluasi terhadap data transaksional membuktikan adanya penumpukan volume pesanan pada variasi menu tertentu. Konsentrasi pesanan yang tidak seimbang ini menegaskan urgensi alokasi stok bahan baku secara asimetris, fokus penuh pada dua puluh menu utama penentu omzet usaha.")

        with col_row1_right:
            st.markdown("### 📌 PB-2: Sebaran Harga Satuan")
            if not df_hv.empty:
                fig2 = px.histogram(df_hv, x="price_satuan", nbins=30, color_discrete_sequence=[C["merah"]])
                fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Harga Satuan (Rp)", yaxis_title="Frekuensi", height=380)
                st.plotly_chart(fig2, use_container_width=True)
                st.info("💡 **Explanatory Insight:** Pemetaan dari 43.227 baris data harga valid memperlihatkan konsentrasi kurva frekuensi yang menumpuk padat di bawah batas 50.000 rupiah. Angka median ini memberikan jangkar kalkulasi riil bagi manajemen dalam menentukan batas atas modal operasional harian.")
            else:
                st.warning("Tidak ada data harga valid untuk filter saat ini.")

        st.markdown("---")
        col_row2_left, col_row2_right = st.columns(2, gap="medium")

        with col_row2_left:
            st.markdown("### 📌 PB-3: Distribusi Segmentasi Harga")
            segmen_counts = df_filtered["segmen_harga"].value_counts().reset_index()
            segmen_counts.columns = ["Segmen Harga", "Jumlah"]
            segmen_counts["sort_idx"] = segmen_counts["Segmen Harga"].apply(lambda x: URUTAN_SEGMEN.index(x) if x in URUTAN_SEGMEN else 99)
            segmen_counts = segmen_counts.sort_values("sort_idx")
            fig3 = px.bar(segmen_counts, x="Segmen Harga", y="Jumlah", color="Segmen Harga", color_discrete_map={
                "Murah (< 20rb)": C["hijau"], "Sedang (20rb-50rb)": C["kuning"], "Agak Mahal (50rb-100rb)": C["biru"], "Mahal (>= 100rb)": C["merah"], "Tidak Disebutkan / Outlier": C["abu"]
            })
            fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False, height=380)
            st.plotly_chart(fig3, use_container_width=True)
            st.info("💡 **Explanatory Insight:** Kluster harga murah dan sedang mendominasi mayoritas mutlak antrean transaksi. Kenyataan ini membuktikan profil konsumen aktif memiliki sensitivitas harga yang tinggi, mengarahkan pemilik toko untuk mengambil opsi paket bundling volume daripada menaikkan margin eceran.")

        with col_row2_right:
            st.markdown("### 📌 PB-4: Proporsi Penyebutan Harga Eksplisit")
            eksplisit_counts = df_filtered["harga_eksplisit"].value_counts().reset_index()
            eksplisit_counts.columns = ["Tipe", "Jumlah"]
            eksplisit_counts["Tipe"] = eksplisit_counts["Tipe"].map({True: "Harga Disebutkan", False: "Harga Tidak Disebutkan"})
            fig4 = px.pie(eksplisit_counts, names="Tipe", values="Jumlah", color="Tipe", color_discrete_map={"Harga Disebutkan": C["biru"], "Harga Tidak Disebutkan": C["abu"]}, hole=0.4)
            fig4.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
            st.plotly_chart(fig4, use_container_width=True)
            st.info("💡 **Explanatory Insight:** Rekam log membuktikan sebagian besar pesan masuk mengabaikan pencantuman harga barang secara jelas. Temuan ini menegaskan bahwa model kecerdasan buatan wajib mengintegrasikan modul pencarian silang otomatis ke database menu internal, menolak ketergantungan penuh pada teks kasir.")

        st.markdown("---")
        col_row3_left, col_row3_right = st.columns(2, gap="medium")

        with col_row3_left:
            st.markdown("### 📌 PB-5: Top 20 Kata Slang Paling Aktif")
            counter_slang, _ = hitung_slang_aktif()
            top_slang = pd.DataFrame(counter_slang.most_common(20), columns=["Slang", "Frekuensi"])
            fig5 = px.bar(top_slang, x="Frekuensi", y="Slang", orientation="h", color_discrete_sequence=[C["kuning"]])
            fig5.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=380)
            st.plotly_chart(fig5, use_container_width=True)
            st.info("💡 **Explanatory Insight:** Singkatan dan istilah informal menduduki peringkat teratas dalam pola ketikan kasir sehari-hari. Integrasi pasokan data dari 1.231 entri slang utama terbukti mampu memotong risiko kegagalan pemrosesan bahasa alami di terminal kasir digital.")

        with col_row3_right:
            st.markdown("### 📌 PB-6: Distribusi Tipe Kuantitas")
            qty_counts = df_filtered["qty_numerik"].value_counts().reset_index()
            qty_counts.columns = ["Tipe Kuantitas", "Jumlah"]
            qty_counts["Tipe Kuantitas"] = qty_counts["Tipe Kuantitas"].map({True: "Numerik (Angka)", False: "Non-Numerik (Teks)"})
            fig6 = px.bar(qty_counts, x="Tipe Kuantitas", y="Jumlah", color="Tipe Kuantitas", color_discrete_map={"Numerik (Angka)": C["hijau"], "Non-Numerik (Teks)": C["merah"]})
            fig6.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False, height=380)
            st.plotly_chart(fig6, use_container_width=True)
            st.info("💡 **Explanatory Insight:** Penulisan kuantitas pesanan menggunakan format alfabet non-numerik masih konsisten muncul di dalam sistem. Pembangunan komponen penerjemah kata sebelum data menyentuh model inti menjadi langkah pengamanan wajib guna menghindari kegagalan kalkulasi final.")

        st.markdown("---")
        st.markdown("### 📌 PB-7: Dominasi Kategori Masakan di Database Produk")
        df_food = muat_makanan()
        kat_counts = df_food["kategori"].value_counts().reset_index()
        kat_counts.columns = ["Kategori", "Jumlah Produk"]
        fig7 = px.bar(kat_counts, x="Jumlah Produk", y="Kategori", orientation="h", color_discrete_sequence=[C["biru"]])
        fig7.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig7, use_container_width=True)
        st.info("💡 **Explanatory Insight:** Inventarisasi 18.558 manifes makanan memperlihatkan penumpukan variasi produk pada segmen masakan tertentu. Ketidakseimbangan representasi data ini memicu risiko bias pengenalan teks, sehingga penambahan sampel kalimat baru untuk kelompok kategori minoritas dilakukan.")

    else:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")


# ==============================================================================
# BAGIAN 5 - HALAMAN 3: SIMULASI CHATKASIR
# ==============================================================================

elif halaman == "💬 Simulasi ChatKasir":

    st.title("💬 Simulasi Ekstraksi Pesanan")
    st.caption("Sistem akan menormalisasi slang lalu mengekstrak entitas secara otomatis.")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("✍️ Teks Obrolan Pelanggan")
        teks_contoh = """[29/5, 07.14] +62 811-2222-3333: order paket ayam bakar madu 10 pack
[29/5, 07.21] Warung Sejahtera: siap harganya 35k"""
        teks_input = st.text_area(label="Ketik teks obrolan di sini:", value=teks_contoh, height=200, label_visibility="collapsed")
        tombol = st.button("🚀 Proses & Ekstrak Entitas", type="primary", use_container_width=True)

    with col_output:
        st.subheader("📤 Hasil Ekstraksi Model AI")

        if not tombol:
            st.markdown("<div style='background:#f0f4f8; border-radius:10px; padding:50px 30px; text-align:center; color:#888; min-height:200px;'>⬅️ Tekan Tombol Proses</div>", unsafe_allow_html=True)
        else:
            with st.spinner("⚙️ Menghubungi Otak AI di Hugging Face Spaces..."):
                import requests
                import json
                
                # Konfigurasi Endpoint API menembak langsung ke Hugging Face Space
                API_URL = "https://achmadrifan-chatkasir.hf.space/predict" 
                
                # Menggunakan label raw_text sesuai spesifikasi payload pada API
                payload = {"raw_text": teks_input}
                
                # Menambahkan kunci akses autentikasi yang diminta oleh server
                headers = {
                    "X-API-Key": "changeme",
                    "Content-Type": "application/json"
                }
                
                try:
                    # Menyisipkan parameter headers ke dalam request
                    response = requests.post(API_URL, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        hasil = response.json()
                        st.success("✅ Ekstraksi Berhasil!")
                        st.code(json.dumps(hasil, ensure_ascii=False, indent=2), language="json")
                        
                        st.markdown("**📋 Ringkasan Pesanan:**")
                        st.dataframe(pd.DataFrame(hasil), use_container_width=True)
                        
                    elif response.status_code == 422:
                        st.error("❌ Validasi Gagal: Format payload tidak sesuai dengan skema API (Unprocessable Entity).")
                        st.json(response.json())
                    elif response.status_code == 400:
                        st.error("❌ Validasi Gagal: Teks terlalu pendek (Minimal 5 karakter).")
                    elif response.status_code == 401:
                        st.error("❌ Autentikasi Gagal: API Key salah atau ditolak oleh server.")
                    elif response.status_code == 503:
                        st.error("❌ API Degraded: Model AI di Hugging Face sedang dimuat. Coba lagi dalam 15 detik.")
                    else:
                        st.error(f"❌ Gagal memproses. Kode Error: {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Gagal terhubung ke URL Hugging Face. Pastikan koneksi internet Anda aktif dan Space dalam keadaan 'Running'.")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan sistem: {e}")
