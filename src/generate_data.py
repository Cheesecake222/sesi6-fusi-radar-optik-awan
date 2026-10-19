"""Membuat data sintetik penyebab gangguan hutan dengan dan tanpa tutupan awan.

Semua data di sini fiktif. Tiap sampel mewakili satu peringatan deforestasi
(deforestation alert) yang penyebabnya sudah diketahui, dengan enam nilai:
dua dari radar Sentinel-1 dan empat dari optik Sentinel-2.

Jalankan:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

from config import (DATA_DIR, FITUR_GABUNGAN, FITUR_OPTIK, JUMLAH_LATIH, JUMLAH_UJI,
                    PERSEN_AWAN_LATIH, SEED, TUTUPAN_AWAN_MAKS, TUTUPAN_AWAN_MIN)

KELAS = ["pertanian", "tambang", "penebangan_selektif", "kebakaran"]
PROPORSI_KELAS = [0.35, 0.15, 0.25, 0.25]

# Rata-rata dan simpangan baku tiap fitur untuk tiap kelas. Kelas sengaja dibuat
# saling tumpang tindih supaya tidak ada yang terlalu mudah ditebak.
CIRI_KELAS = {
    "pertanian":           {"s1_vv": (-2.5, 1.6), "s1_vh": (-3.5, 1.8), "s2_biru": (0.06, 0.025), "s2_hijau": (0.10, 0.03), "s2_merah": (0.16, 0.05), "s2_nir": (0.26, 0.07)},
    "tambang":             {"s1_vv": (0.5, 1.8),  "s1_vh": (-1.0, 1.8), "s2_biru": (0.16, 0.05),  "s2_hijau": (0.20, 0.05), "s2_merah": (0.25, 0.06), "s2_nir": (0.27, 0.07)},
    "penebangan_selektif": {"s1_vv": (-1.0, 1.5), "s1_vh": (-1.8, 1.6), "s2_biru": (0.04, 0.015), "s2_hijau": (0.07, 0.02), "s2_merah": (0.06, 0.025), "s2_nir": (0.36, 0.06)},
    "kebakaran":           {"s1_vv": (-2.0, 1.7), "s1_vh": (-3.2, 1.8), "s2_biru": (0.03, 0.015), "s2_hijau": (0.04, 0.02), "s2_merah": (0.07, 0.03), "s2_nir": (0.13, 0.05)},
}

# Reflektansi awan tebal pada tiap band optik (awan tampak sangat terang)
REFLEKTANSI_AWAN = {"s2_biru": 0.70, "s2_hijau": 0.68, "s2_merah": 0.65, "s2_nir": 0.60}


def buat_permukaan(rng, jumlah):
    """Buat sampel tanpa awan: kelas diundi, lalu tiap fitur diambil dari ciri kelasnya."""
    kelas = rng.choice(KELAS, size=jumlah, p=PROPORSI_KELAS)
    data = {"kelas": kelas}
    for fitur in FITUR_GABUNGAN:
        nilai = np.empty(jumlah)
        for nama_kelas in KELAS:
            mask = kelas == nama_kelas
            rata, sd = CIRI_KELAS[nama_kelas][fitur]
            nilai[mask] = rng.normal(rata, sd, size=mask.sum())
        if fitur in FITUR_OPTIK:
            nilai = np.clip(nilai, 0.0, 1.0)
        data[fitur] = nilai
    return pd.DataFrame(data)


def tambah_awan(df, rng):
    """Kotori nilai optik dengan awan. Radar tidak terpengaruh awan.

    Nilai optik yang teramati = (1 - f) * nilai asli + f * nilai awan,
    dengan f adalah pecahan citra yang tertutup awan.
    """
    hasil = df.copy()
    f = rng.uniform(TUTUPAN_AWAN_MIN, TUTUPAN_AWAN_MAKS, size=len(df))
    for fitur in FITUR_OPTIK:
        awan = rng.normal(REFLEKTANSI_AWAN[fitur], 0.06, size=len(df))
        hasil[fitur] = np.clip((1 - f) * df[fitur].to_numpy() + f * awan, 0.0, 1.0)
    return hasil


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    # Data latih: sebagian sampel berawan, sebagian bersih
    latih = buat_permukaan(rng, JUMLAH_LATIH)
    berawan = rng.random(JUMLAH_LATIH) < PERSEN_AWAN_LATIH
    latih_berawan = tambah_awan(latih, rng)
    latih.loc[berawan, FITUR_OPTIK] = latih_berawan.loc[berawan, FITUR_OPTIK]
    latih.insert(0, "id_sampel", [f"L{i + 1:05d}" for i in range(JUMLAH_LATIH)])
    latih["berawan"] = berawan.astype(int)

    # Data uji: sampel yang sama dibuat dua versi (bersih dan berawan),
    # supaya skenario tutupan awan bisa dibandingkan secara adil
    uji_bersih = buat_permukaan(rng, JUMLAH_UJI)
    uji_berawan = tambah_awan(uji_bersih, rng)
    ids = [f"U{i + 1:05d}" for i in range(JUMLAH_UJI)]
    uji_bersih.insert(0, "id_sampel", ids)
    uji_berawan.insert(0, "id_sampel", ids)

    latih.round(5).to_csv(DATA_DIR / "data_latih.csv", index=False)
    uji_bersih.round(5).to_csv(DATA_DIR / "data_uji_bersih.csv", index=False)
    uji_berawan.round(5).to_csv(DATA_DIR / "data_uji_berawan.csv", index=False)

    print("Data sintetik dibuat di folder data/:")
    print(f"  data_latih.csv        : {len(latih)} sampel ({berawan.mean():.0%} berawan)")
    print(f"  data_uji_bersih.csv   : {len(uji_bersih)} sampel")
    print(f"  data_uji_berawan.csv  : {len(uji_berawan)} sampel (sampel sama, optik tertutup awan)")


if __name__ == "__main__":
    main()
