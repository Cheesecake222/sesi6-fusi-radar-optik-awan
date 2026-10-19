"""Pengaturan parameter analisis.

Semua angka yang bisa diubah ada di file ini, jadi tidak perlu
mengutak-atik kode di file lain.
"""
from pathlib import Path

# --- Lokasi folder ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HASIL_DIR = ROOT / "hasil"

# --- Fitur ------------------------------------------------------------------
# Radar (Sentinel-1): perubahan backscatter sebelum dan sesudah kejadian, dalam dB
FITUR_RADAR = ["s1_vv", "s1_vh"]
# Optik (Sentinel-2): reflektansi setelah kejadian, skala 0 sampai 1
FITUR_OPTIK = ["s2_biru", "s2_hijau", "s2_merah", "s2_nir"]
FITUR_GABUNGAN = FITUR_RADAR + FITUR_OPTIK

# --- Pembuatan data sintetik ----------------------------------------------
SEED = 42
JUMLAH_LATIH = 4000
JUMLAH_UJI = 2000
# Porsi data latih yang optiknya tertutup awan. Di daerah tropis awan sering
# menutupi citra, jadi setengah data latih dibuat berawan.
PERSEN_AWAN_LATIH = 0.5
# Tutupan awan pada citra berawan: pecahan citra yang tertutup
# diambil acak antara dua nilai ini
TUTUPAN_AWAN_MIN = 0.7
TUTUPAN_AWAN_MAKS = 1.0

# --- Evaluasi ---------------------------------------------------------------
# Persentase sampel uji yang optiknya tertutup awan, satu skenario per angka
TINGKAT_AWAN_UJI = [0, 25, 50, 75, 100]
JUMLAH_POHON = 200
ULANGAN_PERMUTASI = 10
