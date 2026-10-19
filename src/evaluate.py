"""Melatih tiga model dan mengevaluasinya pada berbagai tingkat tutupan awan."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import (FITUR_GABUNGAN, FITUR_OPTIK, FITUR_RADAR, JUMLAH_POHON,
                    TINGKAT_AWAN_UJI, ULANGAN_PERMUTASI)

# Tiga skenario masukan. Algoritmanya sama (Random Forest), hanya fitur yang berbeda.
MODEL_FITUR = {
    "Radar saja": FITUR_RADAR,
    "Optik saja": FITUR_OPTIK,
    "Radar + optik": FITUR_GABUNGAN,
}


def latih_model(latih, seed):
    """Latih satu model per skenario. Model tidak diberi tahu mana sampel yang berawan."""
    model = {}
    for nama, fitur in MODEL_FITUR.items():
        rf = RandomForestClassifier(n_estimators=JUMLAH_POHON, random_state=seed, n_jobs=-1)
        rf.fit(latih[fitur], latih["kelas"])
        model[nama] = rf
    return model


def buat_uji_campuran(uji_bersih, uji_berawan, persen, urutan):
    """Ambil sampel uji dengan `persen`% di antaranya memakai versi berawan.

    `urutan` adalah urutan acak tetap, jadi sampel yang berawan pada 25%
    juga masih berawan pada 50% dan seterusnya.
    """
    jumlah = int(round(len(uji_bersih) * persen / 100))
    x = uji_bersih[FITUR_GABUNGAN].to_numpy().copy()
    x_awan = uji_berawan[FITUR_GABUNGAN].to_numpy()
    terpilih = urutan[:jumlah]
    x[terpilih] = x_awan[terpilih]
    return pd.DataFrame(x, columns=FITUR_GABUNGAN)


def evaluasi_tingkat_awan(model, uji_bersih, uji_berawan, seed):
    """Akurasi dan F1 (macro) tiap model pada tiap tingkat tutupan awan."""
    rng = np.random.default_rng(seed)
    urutan = rng.permutation(len(uji_bersih))
    y = uji_bersih["kelas"].to_numpy()

    baris = []
    for persen in TINGKAT_AWAN_UJI:
        x = buat_uji_campuran(uji_bersih, uji_berawan, persen, urutan)
        for nama, fitur in MODEL_FITUR.items():
            prediksi = model[nama].predict(x[fitur])
            baris.append({
                "model": nama,
                "persen_sampel_berawan": persen,
                "akurasi": accuracy_score(y, prediksi),
                "f1_macro": f1_score(y, prediksi, average="macro"),
            })
    return pd.DataFrame(baris)


def prediksi_awan_penuh(model, uji_bersih, uji_berawan):
    """Prediksi tiap model bila semua sampel uji berawan (untuk confusion matrix)."""
    x = uji_berawan[FITUR_GABUNGAN]
    return {nama: model[nama].predict(x[fitur]) for nama, fitur in MODEL_FITUR.items()}


def ketergantungan_sumber(model_gabungan, uji_bersih, uji_berawan, seed):
    """Seberapa bergantung model gabungan pada radar dan pada optik.

    Seluruh kolom satu sumber diacak bersama-sama (permutasi), lalu dihitung
    seberapa besar akurasi turun. Makin besar penurunannya, makin bergantung
    model pada sumber itu. Dilakukan pada data tanpa awan dan data berawan penuh.
    """
    rng = np.random.default_rng(seed)
    y = uji_bersih["kelas"].to_numpy()
    kondisi = {"Tanpa awan": uji_bersih, "Awan penuh": uji_berawan}
    sumber = {"Radar": FITUR_RADAR, "Optik": FITUR_OPTIK}

    baris = []
    for nama_kondisi, data in kondisi.items():
        x = data[FITUR_GABUNGAN].reset_index(drop=True)
        dasar = accuracy_score(y, model_gabungan.predict(x))
        for nama_sumber, kolom in sumber.items():
            penurunan = []
            for _ in range(ULANGAN_PERMUTASI):
                x_acak = x.copy()
                urutan = rng.permutation(len(x))
                x_acak[kolom] = x[kolom].to_numpy()[urutan]
                penurunan.append(dasar - accuracy_score(y, model_gabungan.predict(x_acak)))
            baris.append({
                "kondisi": nama_kondisi,
                "sumber_diacak": nama_sumber,
                "akurasi_dasar": dasar,
                "penurunan_akurasi": float(np.mean(penurunan)),
            })
    return pd.DataFrame(baris)
