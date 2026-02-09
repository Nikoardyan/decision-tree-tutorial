# Decision Tree Tutorial

Proyek ini adalah tutorial sederhana untuk memahami dan membangun model **Decision Tree** menggunakan dataset Iris dari pustaka `scikit-learn`. Decision Tree adalah salah satu algoritma pembelajaran mesin yang paling populer karena mudah dipahami dan diimplementasikan.

---

## Apa itu Decision Tree?

**Decision Tree** adalah algoritma pembelajaran mesin berbasis pohon yang digunakan untuk tugas **klasifikasi** dan **regresi**. Algoritma ini bekerja dengan membagi dataset menjadi subset berdasarkan fitur tertentu, sehingga menghasilkan struktur seperti pohon. Setiap cabang mewakili keputusan, dan setiap daun mewakili hasil atau label.

### Cara Kerja Decision Tree

1. **Pemilihan Fitur Terbaik**:
   - Decision Tree memilih fitur terbaik untuk membagi data berdasarkan kriteria tertentu, seperti:
     - **Gini Impurity**: Mengukur ketidakpastian data dalam sebuah node.
     - **Entropy**: Mengukur ketidakteraturan data dalam sebuah node.
   - Fitur dengan nilai terbaik akan dipilih untuk membagi data.

2. **Pembuatan Node dan Cabang**:
   - Setelah fitur terbaik dipilih, data dibagi menjadi beberapa subset berdasarkan nilai fitur tersebut. Setiap subset menjadi cabang baru dari pohon.

3. **Rekursi**:
   - Proses pembagian data diulang untuk setiap cabang hingga mencapai kondisi tertentu, seperti:
     - Semua data dalam cabang memiliki label yang sama.
     - Tidak ada fitur yang tersisa untuk dibagi.
     - Kedalaman maksimum pohon tercapai.

4. **Prediksi**:
   - Untuk memprediksi, data baru akan melewati pohon dari akar ke daun berdasarkan aturan pembagian yang telah dibuat. Hasil prediksi adalah label yang sesuai dengan daun tempat data tersebut berakhir.

---

## Langkah-langkah Membangun Decision Tree

Berikut adalah langkah-langkah untuk membangun model Decision Tree:

1. **Persiapan Data**:
   - Dataset diimpor dan diproses menjadi dua bagian utama:
     - **Fitur (X)**: Variabel independen yang digunakan untuk membuat prediksi.
     - **Target (y)**: Variabel dependen yang ingin diprediksi.

2. **Pembagian Data**:
   - Dataset dibagi menjadi dua bagian:
     - **Data Latih**: Digunakan untuk melatih model.
     - **Data Uji**: Digunakan untuk menguji performa model.

3. **Membuat Model**:
   - Model Decision Tree dibuat menggunakan pustaka `scikit-learn`.

4. **Pelatihan Model**:
   - Model dilatih menggunakan data latih untuk mempelajari pola dari data.

5. **Prediksi**:
   - Model digunakan untuk memprediksi data uji.

6. **Evaluasi**:
   - Akurasi model dihitung untuk mengevaluasi seberapa baik model dapat memprediksi data baru.

---

## Kode dengan Penjelasan

Berikut adalah kode Python untuk membangun model Decision Tree, dilengkapi dengan komentar untuk menjelaskan setiap langkah:

```python
# filepath: /Users/celinezafirameka/Documents/GitHub/decision-tree-tutorial/decission-tree.py

# Mengabaikan peringatan agar output lebih bersih
import warnings
warnings.filterwarnings("ignore")

# Import library yang dibutuhkan
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Persiapan Data
# Memuat dataset Iris dari scikit-learn
iris = datasets.load_iris()

# Mengonversi dataset menjadi DataFrame untuk mempermudah manipulasi data
data = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],  # Menggabungkan fitur dan target
    columns=iris['feature_names'] + ['target']  # Menambahkan nama kolom
)

# Memisahkan fitur (X) dan target (y)
X = data.drop('target', axis=1)  # Fitur (panjang dan lebar kelopak, dsb.)
y = data['target']  # Target (spesies bunga)

# 2. Pembagian Data
# Membagi data menjadi data latih (70%) dan data uji (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Fitur
    y,  # Target
    test_size=0.3,  # 30% data untuk pengujian
    random_state=42  # Untuk hasil yang konsisten
)

# 3. Membuat Model
# Membuat model Decision Tree dengan parameter default
model = DecisionTreeClassifier()

# 4. Pelatihan Model
# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# 5. Prediksi
# Menggunakan model untuk memprediksi data uji
y_pred = model.predict(X_test)

# 6. Evaluasi
# Menghitung akurasi model dengan membandingkan hasil prediksi dan data sebenarnya
print("Akurasi:", accuracy_score(y_test, y_pred))
```

---

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

---

## Catatan Tambahan

- Dataset **Iris** adalah dataset bawaan `scikit-learn` yang sering digunakan untuk pembelajaran mesin. Dataset ini berisi informasi tentang tiga jenis bunga iris: *setosa*, *versicolor*, dan *virginica*.
- Model Decision Tree yang digunakan adalah model dasar tanpa parameter tambahan. Anda dapat menyesuaikan parameter seperti:
  - `criterion`: Mengatur kriteria pemilihan fitur terbaik (default: `gini`).
  - `max_depth`: Membatasi kedalaman maksimum pohon.
  - `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi node.

---

## Struktur Proyek

```
decision-tree-tutorial/
│
├── decission-tree.py   # File utama untuk melatih model Decision Tree
├── README.md           # Dokumentasi proyek
```

Selamat belajar dan semoga sukses!