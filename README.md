# Decision Tree Tutorial

Proyek ini adalah tutorial sederhana untuk membuat dan melatih model Decision Tree menggunakan dataset Iris dari pustaka `scikit-learn`.

## Cara Kerja Decision Tree

Decision Tree adalah algoritma pembelajaran mesin berbasis pohon yang digunakan untuk tugas klasifikasi dan regresi. Cara kerjanya adalah sebagai berikut:

1. **Pemilihan Fitur**:
   - Decision Tree memilih fitur terbaik untuk membagi data berdasarkan kriteria tertentu, seperti *Gini Impurity* atau *Entropy* (dalam kasus ini, kriteria default adalah *Gini*).

2. **Pembuatan Node**:
   - Data dibagi menjadi beberapa subset berdasarkan fitur yang dipilih. Setiap subset menjadi cabang baru dari pohon.

3. **Rekursi**:
   - Proses pembagian data diulang untuk setiap cabang hingga mencapai kondisi tertentu, seperti:
     - Semua data dalam cabang memiliki label yang sama.
     - Tidak ada fitur yang tersisa untuk dibagi.
     - Kedalaman maksimum pohon tercapai.

4. **Prediksi**:
   - Untuk memprediksi, data baru akan melewati pohon dari akar ke daun berdasarkan aturan pembagian yang telah dibuat.

## Langkah-langkah Membangun Decision Tree

1. **Persiapan Data**:
   - Dataset diimpor dan diproses menjadi fitur (X) dan target (y).

2. **Pembagian Data**:
   - Data dibagi menjadi data latih dan data uji menggunakan `train_test_split`.

3. **Membuat Model**:
   - Model Decision Tree dibuat menggunakan `DecisionTreeClassifier` dari pustaka `scikit-learn`.

4. **Pelatihan Model**:
   - Model dilatih menggunakan data latih.

5. **Prediksi**:
   - Model digunakan untuk memprediksi data uji.

6. **Evaluasi**:
   - Akurasi model dihitung menggunakan metrik seperti `accuracy_score`.

## Komentar dalam Kode

Berikut adalah kode dengan komentar tambahan untuk menjelaskan setiap langkah:

```python
# filepath: /Users/celinezafirameka/Documents/GitHub/decision-tree-tutorial/decission-tree.py
import warnings
warnings.filterwarnings("ignore")  # Mengabaikan peringatan agar output lebih bersih

import numpy as np
import pandas as pd

# Import library dari scikit-learn
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Persiapan Data
iris = datasets.load_iris()  # Memuat dataset Iris bawaan scikit-learn
data = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],  # Menggabungkan fitur dan target
    columns=iris['feature_names'] + ['target']  # Menambahkan nama kolom
)

# Memisahkan fitur (X) dan target (y)
X = data.drop('target', axis=1)  # Fitur (panjang dan lebar kelopak, dsb.)
y = data['target']  # Target (spesies bunga)

# 2. Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Fitur
    y,  # Target
    test_size=0.3,  # 30% data untuk pengujian, 70% untuk pelatihan
    random_state=42  # Untuk hasil yang konsisten
)

# 3. Membuat Model
model = DecisionTreeClassifier()  # Membuat model Decision Tree

# 4. Pelatihan Model
model.fit(X_train, y_train)  # Melatih model dengan data latih

# 5. Prediksi
y_pred = model.predict(X_test)  # Memprediksi data uji

# 6. Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))  # Menghitung akurasi model
```

## Cara Menjalankan

1. Pastikan Anda memiliki Python 3.x dan pustaka berikut terinstal:
   - `numpy`
   - `pandas`
   - `scikit-learn`

2. Jalankan perintah berikut di terminal untuk menjalankan skrip:

   ```bash
   python decission-tree.py
   ```

3. Hasil akurasi model akan ditampilkan di terminal.

## Catatan

- Dataset Iris adalah dataset bawaan `scikit-learn` yang sering digunakan untuk pembelajaran mesin.
- Model Decision Tree yang digunakan adalah model dasar tanpa parameter tambahan. Anda dapat menyesuaikan parameter seperti `max_depth`, `criterion`, atau `min_samples_split` untuk meningkatkan performa model.

## Struktur Proyek

```
decision-tree-tutorial/
│
├── decission-tree.py   # File utama untuk melatih model Decision Tree
├── README.md           # Dokumentasi proyek
```

Selamat mencoba!