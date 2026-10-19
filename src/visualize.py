"""Grafik hasil: akurasi vs awan, confusion matrix, dan ketergantungan pada sumber."""
import matplotlib

matplotlib.use("Agg")  # supaya jalan tanpa jendela grafik
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

WARNA = {"Radar saja": "#3b6fb6", "Optik saja": "#d9822b", "Radar + optik": "#5a9e6f"}


def grafik_akurasi_awan(hasil, akurasi_mayoritas, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for nama, warna in WARNA.items():
        d = hasil[hasil["model"] == nama]
        ax.plot(d["persen_sampel_berawan"], d["akurasi"], marker="o", color=warna, label=nama, linewidth=2)
    ax.axhline(akurasi_mayoritas, color="gray", linestyle="--", linewidth=1,
               label=f"Menebak kelas terbanyak ({akurasi_mayoritas:.0%})")
    ax.set_xlabel("Persen sampel uji yang optiknya tertutup awan")
    ax.set_ylabel("Akurasi")
    ax.set_title("Akurasi klasifikasi penyebab gangguan hutan menurut tutupan awan")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(hasil["persen_sampel_berawan"].unique()))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grafik_confusion(prediksi, y_benar, kelas, path):
    fig, axes = plt.subplots(1, len(prediksi), figsize=(5 * len(prediksi), 4.6))
    for ax, (nama, y_pred) in zip(np.atleast_1d(axes), prediksi.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_benar, y_pred, labels=kelas, normalize="true", cmap="Blues",
            ax=ax, colorbar=False, values_format=".2f", xticks_rotation=35,
        )
        ax.set_title(nama)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Kelas sebenarnya")
    fig.suptitle("Confusion matrix bila semua citra optik tertutup awan (dinormalisasi per kelas)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grafik_ketergantungan(df, path):
    kondisi = ["Tanpa awan", "Awan penuh"]
    sumber = ["Radar", "Optik"]
    warna = {"Radar": "#3b6fb6", "Optik": "#d9822b"}
    posisi = np.arange(len(kondisi))
    lebar = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, s in enumerate(sumber):
        nilai = [df[(df["kondisi"] == k) & (df["sumber_diacak"] == s)]["penurunan_akurasi"].iloc[0] for k in kondisi]
        batang = ax.bar(posisi + (i - 0.5) * lebar, nilai, lebar, label=f"{s} diacak", color=warna[s])
        ax.bar_label(batang, fmt="%.2f", fontsize=9, padding=2)
    ax.set_xticks(posisi)
    ax.set_xticklabels(kondisi)
    ax.set_ylabel("Penurunan akurasi")
    ax.set_title("Model radar + optik: sumber mana yang diandalkan?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
