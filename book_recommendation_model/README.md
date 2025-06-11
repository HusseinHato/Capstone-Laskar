# 📚 Sistem Rekomendasi Buku

Sistem rekomendasi buku bertenaga AI lengkap yang dibangun menggunakan TF-IDF dan cosine similarity. Sistem ini mencakup pelatihan model machine learning, backend REST API, dan antarmuka web modern.

---

## 🌟 Fitur

* **Rekomendasi Cerdas**: Menggunakan TF-IDF dan cosine similarity
* **Dukungan Multi-Bahasa**: Bahasa Indonesia & Inggris
* **RESTful API**: Backend lengkap dengan endpoint bervariasi
* **Antarmuka Modern**: UI responsif dengan pencarian waktu nyata
* **Pemrosesan Batch**: Latih model dari file Excel
* **Pemantauan Langsung**: Status API & model secara real-time

---

## 🏗️ Arsitektur Sistem

```
book-recommendation-system/
├── main.py                   # Skrip pelatihan model
├── app.py                    # Server Flask API
├── index.html                # Antarmuka Web
├── databuku.xlsx             # Dataset buku
└── book_recommendation_model/  # File model hasil pelatihan
    ├── tfidf_vectorizer.pkl
    ├── tfidf_matrix.pkl
    ├── books_data.json
    └── model_info.json
```

---

## 📋 Persyaratan

### Instalasi Dependensi Python

```bash
pip install pandas numpy scikit-learn flask flask-cors openpyxl
```

### Format File Excel

| Kolom       | Deskripsi          |
| ----------- | ------------------ |
| Author      | Nama penulis buku  |
| Judul\_Buku | Judul buku         |
| Deskripsi   | Deskripsi isi buku |
| Penerbit    | Nama penerbit      |

---

## 🚀 Cara Cepat Memulai

1. **Instal Dependensi**

```bash
pip install pandas numpy scikit-learn flask flask-cors openpyxl
```

2. **Siapkan Dataset**
   Letakkan file Excel (mis. `databuku.xlsx`) di direktori proyek.

3. **Latih Model**

```bash
python main.py
```

Model disimpan di `book_recommendation_model/`.

4. **Jalankan API Server**

```bash
python app.py
```

Akses di: [http://localhost:5000](http://localhost:5000)

5. **Buka Antarmuka Web**
   Buka `index.html` di browser.

---

## 📖 Panduan Penggunaan

### Melatih Model Baru

* Pastikan kolom Excel sesuai
* Edit path file di `main.py`
* Jalankan `python main.py`

### Menggunakan UI Web

* Ketik judul/penulis/kata kunci
* Atur hasil (jumlah & ambang kemiripan)
* Jelajahi seluruh buku dengan paginasi

---

## 🔌 Endpoint API

| Endpoint          | Method | Fungsi                 | Parameter                          |
| ----------------- | ------ | ---------------------- | ---------------------------------- |
| `/`               | GET    | Info API               | -                                  |
| `/health`         | GET    | Cek kesehatan API      | -                                  |
| `/model-info`     | GET    | Info model             | -                                  |
| `/search`         | POST   | Cari buku (kueri JSON) | `query`, `top_k`, `min_similarity` |
| `/search/<query>` | GET    | Cari buku via URL      | `top_k`, `min_similarity`          |
| `/books`          | GET    | Tampilkan seluruh buku | `page`, `per_page`                 |

Contoh:

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5, "min_similarity": 0.1}'
```

---

## 🔧 Konfigurasi

### `main.py`

```python
TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.8,
    lowercase=True
)
```

### `app.py`

```python
app.run(
    host='0.0.0.0',
    port=5000,
    debug=False,
    threaded=True
)
```

---

## 📊 Performa Model

* **Presisi Tinggi**: TF-IDF cocok untuk teks panjang
* **Responsif**: Waktu respon < 100ms
* **Skalabel**: Efisien untuk ribuan data
* **Bahasa**: Mendukung bahasa Indonesia

---

## 🔎 Pengembangan Selanjutnya

* Rekomendasi berbasis pengguna (collaborative filtering)
* Embedding deep learning
* Personalisasi berdasarkan profil
* Sistem penilaian dan filter kategori
* Query boolean dan pencarian frasa
* Caching dengan Redis
* Analitik dan pelacakan penggunaan

---

## 🙏 Terima Kasih

* scikit-learn
* Flask
* TF-IDF
* Desain CSS modern

**Dibuat dengan ❤️ untuk para pecinta buku dan data!**
