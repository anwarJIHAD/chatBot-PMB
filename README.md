
# Chatbot Pendaftaran Mahasiswa Baru (PMB) Berbasis Machine Learning

Proyek ini adalah implementasi chatbot berbasis Machine Learning untuk menjawab pertanyaan-pertanyaan umum terkait Pendaftaran Mahasiswa Baru (PMB) di Politeknik Caltex Riau (PCR). Chatbot dibangun menggunakan Python, Flask, dan Scikit-learn.

---

## 🎯 Tujuan

Membantu calon mahasiswa mendapatkan informasi seputar PMB secara cepat dan efisien dengan memanfaatkan teknologi AI melalui klasifikasi intent pertanyaan.

---

## 🧠 Teknologi yang Digunakan

- **Python 3.10+**
- **Flask** - Web Framework
- **Scikit-learn** - Pembuatan model Machine Learning
- **Pandas** - Untuk manipulasi dataset
- **TF-IDF Vectorizer** - Untuk representasi teks
- **Naive Bayes Classifier** - Model klasifikasi intent
- **Joblib** - Untuk menyimpan dan memuat model

---

## 📁 Struktur Proyek

```
📦 chatbot-pmb-ml
├── app.py                     # Aplikasi utama Flask
├── intent_model.pkl          # Model klasifikasi intent
├── vectorizer.pkl            # TF-IDF vectorizer
├── dataset/
│   └── intent_dataset.csv    # Dataset pelatihan (pertanyaan + label intent)
├── templates/
│   └── index.html            # Tampilan web chatbot
├── static/
│   └── style.css             # (Opsional) Styling chatbot
└── README.md                 # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Proyek

1. **Clone repository ini**
   ```bash
   git clone https://github.com/username/chatbot-pmb-ml.git
   cd chatbot-pmb-ml
   ```

2. **Buat environment dan install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   pip install -r requirements.txt
   ```

3. **Latih model (jika belum ada)**
   ```bash
   python train_model.py
   ```

4. **Jalankan aplikasi Flask**
   ```bash
   python app.py
   ```

5. **Buka di browser**
   ```
   http://localhost:5000
   ```

---

## 📊 Dataset

Dataset terdiri dari kumpulan pertanyaan PMB yang dikategorikan ke dalam beberapa label intent, misalnya:

| Pertanyaan                          | Intent         |
|------------------------------------|----------------|
| Biaya kuliah di PCR berapa ya?     | biaya          |
| Jalur masuk apa saja yang tersedia?| jalur_pendaftaran |
| Apa saja persyaratan pendaftaran?  | syarat         |
| Jadwal pendaftaran kapan?          | jadwal         |

Dataset dapat dikembangkan lebih lanjut sesuai kebutuhan institusi.

---

## 🧪 Hasil & Akurasi

- Akurasi validasi: **~87%** (dengan dataset awal 150 data)
- Respons chatbot mampu membedakan pertanyaan seputar:
  - Jadwal
  - Biaya
  - Syarat
  - Jalur pendaftaran
  - dll.

---

## 📦 Fitur Utama

✅ Intent Classification dengan ML  
✅ UI berbasis web (Flask + HTML)  
✅ Model ringan dan cepat dilatih  
✅ Bisa dikembangkan lebih lanjut ke platform WhatsApp / Telegram  

---

## 📌 Rencana Pengembangan

- [x] Intent Classification v1
- [x] Web UI sederhana
- [ ] Integrasi ke website resmi pmb.pcr.ac.id
- [ ] Penambahan fitur fallback dan suggestion
- [ ] Perluasan dataset hingga 300+ pertanyaan

---

## 💡 Kontribusi

Pull request sangat diterima! Jika kamu ingin menambahkan intent baru, memperbaiki UI, atau memperluas dataset, silakan fork repo ini dan buat PR.

---

## 📄 Lisensi

MIT License - Silakan digunakan dan dimodifikasi untuk keperluan edukasi dan pengembangan chatbot institusi.

---

## 📬 Kontak

Mhd Anwar  
📧 mhdanware1@gmail.com  
🌐 https://straair.com
