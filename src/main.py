"""Program utama: latih tiga model, uji pada berbagai tingkat awan, simpan hasil.

Jalankan (dari folder utama project):  python src/main.py
"""
import pandas as pd

from config import DATA_DIR, FITUR_GABUNGAN, HASIL_DIR, SEED, TINGKAT_AWAN_UJI
from evaluate import (evaluasi_tingkat_awan, ketergantungan_sumber, latih_model,
                      prediksi_awan_penuh)
from generate_data import KELAS
from visualize import grafik_akurasi_awan, grafik_confusion, grafik_ketergantungan

KOLOM_WAJIB = {"id_sampel", "kelas"} | set(FITUR_GABUNGAN)


def baca_csv(nama_file):
    path = DATA_DIR / nama_file
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} tidak ditemukan. Jalankan dulu: python src/generate_data.py"
        )
    df = pd.read_csv(path)
    kurang = KOLOM_WAJIB - set(df.columns)
    if kurang:
        raise ValueError(f"{nama_file} kekurangan kolom: {sorted(kurang)}")
    if df.empty:
        raise ValueError(f"{nama_file} kosong")
    if df[list(KOLOM_WAJIB)].isna().any().any():
        raise ValueError(f"{nama_file} berisi nilai kosong")
    tidak_dikenal = set(df["kelas"]) - set(KELAS)
    if tidak_dikenal:
        raise ValueError(f"{nama_file} berisi kelas yang tidak dikenal: {sorted(tidak_dikenal)}")
    return df


def main():
    if any(p < 0 or p > 100 for p in TINGKAT_AWAN_UJI):
        raise ValueError("TINGKAT_AWAN_UJI di config.py harus berisi angka 0 sampai 100")

    latih = baca_csv("data_latih.csv")
    uji_bersih = baca_csv("data_uji_bersih.csv")
    uji_berawan = baca_csv("data_uji_berawan.csv")
    if not uji_bersih["id_sampel"].equals(uji_berawan["id_sampel"]):
        raise ValueError("data_uji_bersih.csv dan data_uji_berawan.csv harus berisi sampel yang sama dengan urutan sama")
    if not uji_bersih["kelas"].equals(uji_berawan["kelas"]):
        raise ValueError("Label kelas pada kedua file uji tidak sama")

    model = latih_model(latih, SEED)
    hasil = evaluasi_tingkat_awan(model, uji_bersih, uji_berawan, SEED)
    prediksi = prediksi_awan_penuh(model, uji_bersih, uji_berawan)
    ketergantungan = ketergantungan_sumber(model["Radar + optik"], uji_bersih, uji_berawan, SEED)

    HASIL_DIR.mkdir(exist_ok=True)
    hasil.round(4).to_csv(HASIL_DIR / "akurasi_vs_awan.csv", index=False)
    ketergantungan.round(4).to_csv(HASIL_DIR / "ketergantungan_sumber.csv", index=False)

    akurasi_mayoritas = uji_bersih["kelas"].value_counts(normalize=True).max()
    grafik_akurasi_awan(hasil, akurasi_mayoritas, HASIL_DIR / "akurasi_vs_awan.png")
    grafik_confusion(prediksi, uji_bersih["kelas"].to_numpy(), KELAS, HASIL_DIR / "confusion_awan_penuh.png")
    grafik_ketergantungan(ketergantungan, HASIL_DIR / "ketergantungan_sumber.png")

    print(f"Data latih: {len(latih)} sampel | data uji: {len(uji_bersih)} sampel")
    print(f"Menebak kelas terbanyak: akurasi {akurasi_mayoritas:.3f}")
    print("\nAkurasi menurut persen sampel uji yang berawan:")
    tabel = hasil.pivot(index="persen_sampel_berawan", columns="model", values="akurasi")
    print(tabel[["Radar saja", "Optik saja", "Radar + optik"]].round(3).to_string())
    print("\nF1 macro:")
    tabel_f1 = hasil.pivot(index="persen_sampel_berawan", columns="model", values="f1_macro")
    print(tabel_f1[["Radar saja", "Optik saja", "Radar + optik"]].round(3).to_string())
    print("\nPenurunan akurasi model gabungan bila satu sumber diacak:")
    print(ketergantungan.round(3).to_string(index=False))
    print(f"\nHasil tersimpan di folder: {HASIL_DIR}")


if __name__ == "__main__":
    main()
