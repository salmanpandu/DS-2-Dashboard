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

# Konfigurasi halaman dasar tanpa ikon bawaan
st.set_page_config(
    page_title="ChatKasir Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Konstanta warna brand (Tema Hijau UI/UX)
C = {
    "hijau_utama"  : "#2D6A4F",
    "hijau_gelap"  : "#1B4332",
    "hijau_sedang" : "#40916C",
    "hijau_muda"   : "#74C69D",
    "hijau_terang" : "#B7E4C7",
    "abu"          : "#95A5A6",
    "hijau_bg"     : "#E9F5EC",
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

# Komponen UI Khusus untuk Insight Box
def render_insight(teks):
    st.markdown(f"""
    <div style='background-color: {C["hijau_bg"]}; border-left: 5px solid {C["hijau_utama"]}; padding: 16px 20px; border-radius: 4px; color: {C["hijau_gelap"]}; margin: 15px 0; font-size: 0.95rem; line-height: 1.6; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>
        <strong style='font-size: 1rem; display: block; margin-bottom: 5px;'>Explanatory Insight</strong>
        {teks}
    </div>
    """, unsafe_allow_html=True)

# Fungsi pembersihan tampilan Plotly
def terapkan_tema_bersih(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=C["hijau_gelap"]),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False)
    return fig

# ==============================================================================
# BAGIAN 1 - FUNGSI LOAD & CLEANING DATA
# ==============================================================================

@st.cache_data
def muat_sintetis():
    df = pd.read_csv("chatkasir_synthetic.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    df["harga_eksplisit"] = df["price_satuan"] != -1
    df["harga_valid"]     = (df["price_satuan"] >= HARGA_MIN_VALID) & (df["price_satuan"] <= HARGA_MAX_VALID)
    df["harga_outlier"]   = df["harga_eksplisit"] & (df["harga_valid"] == False)
    df["multi_produk"]    = df["product"].str.contains(" & ", na=False)
    df["qty_numerik"]     = pd.to_numeric(df["quantity"], errors="coerce").notna()

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
    df = pd.read_csv("food_utama.csv")
    df["name"]        = df["name"].str.strip().str.lower()
    df["name_clean"]  = df["name"]
    df["kategori"]    = df["name_clean"].apply(lambda n: PETA_KATEGORI.get(n.split()[0], "Lainnya"))
    df["panjang_nama"] = df["name_clean"].apply(lambda n: len(n.split()))
    return df

@st.cache_data
def muat_slang():
    df = pd.read_csv("slang_utama.csv")
    df["slang"]  = df["slang"].str.strip().str.lower()
    df["formal"] = df["formal"].str.strip().str.lower()
    df["panjang_slang"]   = df["slang"].str.len()
    df["panjang_formal"]  = df["formal"].str.len()
    df["selisih_panjang"] = df["panjang_formal"] - df["panjang_slang"]
    df["tipe"] = df["selisih_panjang"].apply(lambda s: "Disingkat" if s > 0 else ("Sama" if s == 0 else "Diperpanjang"))
    return df

@st.cache_data
def hitung_slang_aktif(n_sample: int = 10_000):
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
    df = muat_sintetis()
    mask_pola = df["pattern"].isin(pola_tuple)
    mask_harga = (
        (df["harga_valid"] == False) |
        ((df["price_satuan"] >= harga_min) & (df["price_satuan"] <= harga_max))
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
        f"""
        <h1 style='color:{C["hijau_gelap"]}; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0px;'>ChatKasir</h1>
        <p style='color:{C["hijau_sedang"]}; font-size: 13px; margin-top: 0px; font-weight: 500;'>Analytics Dashboard</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    halaman = st.radio(
        "Navigasi Modul",
        options=[
            "Ringkasan Data",
            "Analisis Bisnis",
            "Simulasi ChatKasir",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if halaman == "Analisis Bisnis":
        st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.1rem;'>Filter Data Interaktif</h3>", unsafe_allow_html=True)

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
            pola_pilihan = [1, 2, 3, 4]

        segmen_pilihan = st.multiselect(
            "Segmen Harga",
            options=URUTAN_SEGMEN,
            default=URUTAN_SEGMEN,
        )
        if not segmen_pilihan:
            segmen_pilihan = URUTAN_SEGMEN

    else:
        harga_min      = 5_000
        harga_max      = 500_000
        pola_pilihan   = [1, 2, 3, 4]
        segmen_pilihan = URUTAN_SEGMEN

    st.markdown("---")
    st.caption("CC26-PSU065 . DBS Foundation")


df_filtered = terapkan_filter(
    harga_min,
    harga_max,
    tuple(sorted(pola_pilihan)),
    tuple(sorted(segmen_pilihan)),
)


# ==============================================================================
# BAGIAN 3 - HALAMAN 1: RINGKASAN DATA
# ==============================================================================

if halaman == "Ringkasan Data":

    st.markdown(f"<h1 style='color:{C['hijau_gelap']};'>Ringkasan Dataset ChatKasir</h1>", unsafe_allow_html=True)
    st.markdown("Ikhtisar pangkalan data utama yang menyuplai mesin kecerdasan buatan, mencakup log percakapan sintetis, manifes produk, dan leksikon slang.")

    df_synth = muat_sintetis()
    df_food  = muat_makanan()
    df_slang = muat_slang()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Data Percakapan Sintetis", f"{len(df_synth):,}")
    k2.metric("Nama Produk Makanan", f"{len(df_food):,}")
    k3.metric("Entri Kamus Slang", f"{len(df_slang):,}")
    k4.metric("Kategori Masakan", f"{df_food['kategori'].nunique()}")

    st.markdown("---")

    st.subheader("Formulasi Pertanyaan Bisnis")
    st.markdown(
        "Rumusan ini mendefinisikan batas evaluasi analitik, menahan proses perburuan data acak, dan memfokuskan ekstraksi wawasan pada tindakan koreksi operasional UMKM yang terukur."
    )
    
    with st.expander("Metode SMART + 4W"):
        st.markdown(
            "Kerangka Kerja SMART memaksa pertanyaan menjadi Spesifik, Terukur, Berorientasi aksi, Relevan, dan Dibatasi waktu. "
            "Integrasi formula 4W (What, Why, Where, When) mempertajam sudut pandang sehingga setiap metrik visual bermuara pada satu keputusan eksekutif."
        )

    pb_data = pd.DataFrame([
        {"Kode": "PB-1", "Fokus": "What", "Pertanyaan": "Apa 20 produk yang paling sering dipesan pelanggan?", "Tindakan Koreksi": "Fokuskan kapasitas suplai pada produk penentu omzet"},
        {"Kode": "PB-2", "Fokus": "What", "Pertanyaan": "Berapa rata-rata & median harga satuan produk eksplisit?", "Tindakan Koreksi": "Tentukan batas harga aman menggunakan metrik median"},
        {"Kode": "PB-3", "Fokus": "What", "Pertanyaan": "Bagaimana distribusi segmentasi harga pasar?", "Tindakan Koreksi": "Eksekusi strategi penjualan silang (bundling) berbasis volume"},
        {"Kode": "PB-4", "Fokus": "Why",  "Pertanyaan": "Mengapa 56% pesanan masuk tanpa harga eksplisit?", "Tindakan Koreksi": "Bangun jalur silang otomatis ke pangkalan harga internal"},
        {"Kode": "PB-5", "Fokus": "What", "Pertanyaan": "Apa kata slang yang mendominasi percakapan?", "Tindakan Koreksi": "Kalibrasi mesin pengenalan bahasa pada kosakata utama"},
        {"Kode": "PB-6", "Fokus": "What", "Pertanyaan": "Bagaimana sebaran penulisan kuantitas pesanan?", "Tindakan Koreksi": "Terapkan modul pengubah teks alfabetikal menjadi angka mutlak"},
        {"Kode": "PB-7", "Fokus": "What", "Pertanyaan": "Kategori produk apa yang menguasai pangkalan data?", "Tindakan Koreksi": "Suntikkan data latih paksa pada kategori minoritas"},
    ])
    st.dataframe(pb_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Tinjauan Tabel Mentah")
    tab1, tab2, tab3 = st.tabs(["Data Sintetis", "Data Makanan", "Data Slang"])
    with tab1:
        st.dataframe(df_synth.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df_food.head(20), use_container_width=True)
    with tab3:
        st.dataframe(df_slang.head(20), use_container_width=True)


# ==============================================================================
# BAGIAN 4 - HALAMAN 2: ANALISIS BISNIS
# ==============================================================================

elif halaman == "Analisis Bisnis":

    import plotly.express as px

    st.markdown(f"<h1 style='color:{C['hijau_gelap']};'>Analisis Bisnis Terpadu</h1>", unsafe_allow_html=True)
    st.markdown(f"Mengeksekusi pemrosesan waktu nyata pada {len(df_filtered):,} baris data pesanan.")

    df_hv = df_filtered[df_filtered["harga_valid"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Pesanan (Tersaring)", f"{len(df_filtered):,}")
    k2.metric("Rata-rata Harga Satuan", f"Rp {df_hv['price_satuan'].mean():,.0f}" if len(df_hv) else "N/A")
    k3.metric("Median Harga Satuan", f"Rp {df_hv['price_satuan'].median():,.0f}" if len(df_hv) else "N/A")
    k4.metric("Harga Tidak Ditulis", f"{(df_filtered['harga_eksplisit'] == False).sum():,}", delta=f"{(df_filtered['harga_eksplisit'] == False).mean()*100:.1f}%", delta_color="off")

    st.markdown("---")

    if not df_filtered.empty:
        col_row1_left, col_row1_right = st.columns(2, gap="large")
        
        with col_row1_left:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-1: Produk Berkinerja Puncak</h3>", unsafe_allow_html=True)
            top_produk = df_filtered["product"].value_counts().head(20).reset_index()
            top_produk.columns = ["Produk", "Jumlah Pesanan"]
            fig1 = px.bar(top_produk, x="Jumlah Pesanan", y="Produk", orientation="h", color_discrete_sequence=[C["hijau_utama"]])
            fig1 = terapkan_tema_bersih(fig1)
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
            
            render_insight("Konsentrasi volume pesanan menumpuk pada dua puluh menu utama. Mayoritas variasi produk di luar daftar ini membebani rantai pasok. Data ini mendesak penerapan prinsip Occam's Razor dalam manajemen inventaris. Pangkas menu yang memakan biaya simpan tanpa memberikan perputaran modal nyata. Fokuskan seluruh kapasitas dapur untuk mengamankan stok bahan baku para penyumbang omzet utama ini.")

        with col_row1_right:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-2: Sebaran Kepadatan Harga</h3>", unsafe_allow_html=True)
            if not df_hv.empty:
                fig2 = px.histogram(df_hv, x="price_satuan", nbins=30, color_discrete_sequence=[C["hijau_sedang"]])
                fig2 = terapkan_tema_bersih(fig2)
                fig2.update_layout(xaxis_title="Harga Satuan (Rp)", yaxis_title="Frekuensi Kemunculan")
                st.plotly_chart(fig2, use_container_width=True)
                
                render_insight("Rentang harga valid terkunci padat di bawah angka 50.000 rupiah. Kurva distribusi yang condong ke kiri membuktikan sensitivitas dompet pelanggan sangat tajam. Merilis menu berharga premium di tengah ekosistem ini adalah bunuh diri finansial. Angka median berfungsi sebagai batas psikologis. Melewati batas ini berarti menantang daya beli pasar yang sudah terbentuk kuat.")
            else:
                st.info("Rentang data harga valid tidak ditemukan pada pengaturan filter saat ini.")

        st.markdown("---")
        col_row2_left, col_row2_right = st.columns(2, gap="large")

        with col_row2_left:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-3: Segmentasi Harga Dominan</h3>", unsafe_allow_html=True)
            segmen_counts = df_filtered["segmen_harga"].value_counts().reset_index()
            segmen_counts.columns = ["Segmen Harga", "Jumlah"]
            segmen_counts["sort_idx"] = segmen_counts["Segmen Harga"].apply(lambda x: URUTAN_SEGMEN.index(x) if x in URUTAN_SEGMEN else 99)
            segmen_counts = segmen_counts.sort_values("sort_idx")
            fig3 = px.bar(segmen_counts, x="Segmen Harga", y="Jumlah", color="Segmen Harga", color_discrete_map={
                "Murah (< 20rb)": C["hijau_terang"], 
                "Sedang (20rb-50rb)": C["hijau_muda"], 
                "Agak Mahal (50rb-100rb)": C["hijau_sedang"], 
                "Mahal (>= 100rb)": C["hijau_gelap"], 
                "Tidak Disebutkan / Outlier": C["abu"]
            })
            fig3 = terapkan_tema_bersih(fig3)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            
            render_insight("Dominasi absolut berada di segmen harga murah dan sedang. Realitas ekonomi pasar menuntut eksekusi strategi penjualan berbasis volume. Pelanggan mencari batas aman pengeluaran harian. Rancang paket kombo yang menggabungkan menu pendorong lalu lintas dengan produk pelengkap bermargin tinggi. Pendekatan ini memaksa nilai transaksi naik tanpa memicu penolakan dari konsumen.")

        with col_row2_right:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-4: Ketergantungan Kalkulasi Harga</h3>", unsafe_allow_html=True)
            eksplisit_counts = df_filtered["harga_eksplisit"].value_counts().reset_index()
            eksplisit_counts.columns = ["Tipe", "Jumlah"]
            eksplisit_counts["Tipe"] = eksplisit_counts["Tipe"].map({True: "Harga Dituliskan", False: "Harga Diabaikan"})
            fig4 = px.pie(eksplisit_counts, names="Tipe", values="Jumlah", color="Tipe", color_discrete_map={"Harga Dituliskan": C["hijau_sedang"], "Harga Diabaikan": C["hijau_terang"]}, hole=0.45)
            fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=C["hijau_gelap"]), margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig4, use_container_width=True)
            
            render_insight("Sebanyak 56 persen pesanan masuk mengabaikan nominal harga. Pelanggan melempar beban kalkulasi secara penuh kepada operator toko. Fakta ini menghancurkan asumsi bahwa teks masukan cukup untuk memproses transaksi. Otak buatan yang dibangun wajib memiliki jalur silang khusus ke pangkalan data harga internal. Mengandalkan model bahasa untuk menebak harga menciptakan celah kerugian finansial.")

        st.markdown("---")
        col_row3_left, col_row3_right = st.columns(2, gap="large")

        with col_row3_left:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-5: Pemetaan Kosakata Cair</h3>", unsafe_allow_html=True)
            counter_slang, _ = hitung_slang_aktif()
            top_slang = pd.DataFrame(counter_slang.most_common(20), columns=["Slang", "Frekuensi"])
            fig5 = px.bar(top_slang, x="Frekuensi", y="Slang", orientation="h", color_discrete_sequence=[C["hijau_muda"]])
            fig5 = terapkan_tema_bersih(fig5)
            fig5.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig5, use_container_width=True)
            
            render_insight("Interaksi digital dikuasai oleh frasa nonformal dan singkatan lokal. Pelanggan menolak berbicara kaku. Tabel frekuensi membuktikan bahwa mengabaikan normalisasi teks sama dengan membutakan algoritma. Pemetaan ribuan entri slang ini bertindak sebagai fondasi jembatan komunikasi. Tanpanya, akurasi pemrosesan bahasa alami runtuh seketika saat menghadapi teks dunia nyata.")

        with col_row3_right:
            st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-6: Format Hitungan Deskriptif</h3>", unsafe_allow_html=True)
            qty_counts = df_filtered["qty_numerik"].value_counts().reset_index()
            qty_counts.columns = ["Tipe Kuantitas", "Jumlah"]
            qty_counts["Tipe Kuantitas"] = qty_counts["Tipe Kuantitas"].map({True: "Numerik (Angka Mutlak)", False: "Deskriptif (Alfabetikal)"})
            fig6 = px.bar(qty_counts, x="Tipe Kuantitas", y="Jumlah", color="Tipe Kuantitas", color_discrete_map={"Numerik (Angka Mutlak)": C["hijau_gelap"], "Deskriptif (Alfabetikal)": C["hijau_sedang"]})
            fig6 = terapkan_tema_bersih(fig6)
            fig6.update_layout(showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)
            
            render_insight("Konsumen sering mengetikkan jumlah pesanan menggunakan abjad bebas atau pecahan. Mesin tidak memahami teks hitungan secara bawaan. Modul penerjemah teks ke angka mutlak wajib diletakkan pada lapisan terluar pemrosesan. Membiarkan format mentah ini menembus model inti jaringan saraf akan melumpuhkan fungsi kalkulasi matematis sistem mesin kasir.")

        st.markdown("---")
        st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>PB-7: Bias Representasi Menu Masakan</h3>", unsafe_allow_html=True)
        df_food = muat_makanan()
        kat_counts = df_food["kategori"].value_counts().reset_index()
        kat_counts.columns = ["Kategori", "Jumlah Produk"]
        fig7 = px.bar(kat_counts, x="Jumlah Produk", y="Kategori", orientation="h", color_discrete_sequence=[C["hijau_utama"]])
        fig7 = terapkan_tema_bersih(fig7)
        fig7.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig7, use_container_width=True)
        
        render_insight("Inventarisasi belasan ribu nama produk memperlihatkan ketimpangan mencolok pada kategori tertentu. Kondisi ini melahirkan bias laten. Model kecerdasan buatan menjadi sangat tajam mengenali satu jenis menu namun tumpul total pada kategori lain. Anda harus menyuntikkan sampel teks tambahan secara paksa untuk kategori minoritas agar mesin tidak diskriminatif dalam membaca ragam pesanan pelanggan.")

    else:
        st.warning("Pengaturan filter gagal menemukan kecocokan pada pangkalan data.")


# ==============================================================================
# BAGIAN 5 - HALAMAN 3: SIMULASI CHATKASIR
# ==============================================================================

elif halaman == "Simulasi ChatKasir":

    st.markdown(f"<h1 style='color:{C['hijau_gelap']};'>Simulasi Ekstraksi AI</h1>", unsafe_allow_html=True)
    st.markdown("Sistem ini terhubung langsung dengan endpoint arsitektur saraf tiruan untuk mengekstrak entitas pesanan secara otomatis.")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>Masukan Log Percakapan</h3>", unsafe_allow_html=True)
        teks_contoh = """[29/5, 07.14] +62 811-2222-3333: order paket ayam bakar madu 10 pack
[29/5, 07.21] Warung Sejahtera: siap harganya 35k"""
        teks_input = st.text_area(label="Ketik teks obrolan di sini", value=teks_contoh, height=200, label_visibility="collapsed")
        tombol = st.button("Jalankan Ekstraksi Pemrosesan", type="primary", use_container_width=True)

    with col_output:
        st.markdown(f"<h3 style='color:{C['hijau_gelap']}; font-size:1.2rem;'>Hasil Pembacaan Mesin</h3>", unsafe_allow_html=True)

        if not tombol:
            st.markdown(f"<div style='background:{C['hijau_bg']}; border-radius:6px; padding:45px 30px; text-align:center; color:{C['hijau_sedang']}; min-height:200px; border: 1px dashed {C['hijau_muda']};'>Menunggu Perintah Eksekusi</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Menghubungi infrastruktur saraf tiruan di Hugging Face Spaces..."):
                import requests
                import json
                
                API_URL = "https://achmadrifan-chatkasir.hf.space/predict" 
                payload = {"raw_text": teks_input}
                headers = {
                    "X-API-Key": "changeme",
                    "Content-Type": "application/json"
                }
                
                try:
                    response = requests.post(API_URL, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        hasil = response.json()
                        st.markdown(f"<div style='background-color: {C['hijau_bg']}; color: {C['hijau_gelap']}; padding: 12px; border-radius: 4px; margin-bottom: 15px; font-weight: bold;'>Pembacaan Entitas Berhasil Diselesaikan</div>", unsafe_allow_html=True)
                        st.code(json.dumps(hasil, ensure_ascii=False, indent=2), language="json")
                        
                        st.markdown("<p style='font-weight: 600; margin-top: 15px;'>Tabel Hasil Rekapitulasi Pembeli:</p>", unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(hasil), use_container_width=True)
                        
                    elif response.status_code == 422:
                        st.error("Kegagalan Validasi: Format paket data ditolak akibat ketidaksesuaian skema API sistem internal.")
                        st.json(response.json())
                    elif response.status_code == 400:
                        st.error("Kegagalan Parameter: Kepadatan teks obrolan tidak memenuhi batas minimal evaluasi mesin.")
                    elif response.status_code == 401:
                        st.error("Penolakan Akses: Kunci autentikasi peladen gagal diverifikasi oleh sistem pengamanan utama.")
                    elif response.status_code == 503:
                        st.error("Latensi Infrastruktur: Otak jaringan saraf sedang dibangunkan dari status rehat. Kirimkan ulang permintaan.")
                    else:
                        st.error(f"Kegagalan Pemrosesan Kode {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("Kegagalan Jaringan: Jalur akses menuju infrastruktur awan Hugging Face terputus.")
                except Exception as e:
                    st.error(f"Kegagalan Modul Internal: {e}")
