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
        else:             return "Mahal (>= 100
