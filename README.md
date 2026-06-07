# MedScan-AI 🩺

MedScan-AI adalah aplikasi berbasis web yang menggunakan teknologi *Natural Language Processing* (NLP) dan *Machine Learning* untuk memprediksi kemungkinan penyakit berdasarkan gejala yang dialami oleh pengguna. Aplikasi ini menerima input gejala (dalam bahasa Indonesia maupun Inggris), menerjemahkannya jika perlu, dan memberikan diagnosis awal beserta tingkat kepercayaannya.

## ✨ Fitur Utama
- **Prediksi Penyakit Pintar**: Memprediksi penyakit dari model Machine Learning berdasarkan input teks gejala.
- **Dukungan Multi-Bahasa**: Menerima input gejala dalam bahasa Indonesia dan otomatis menerjemahkannya ke bahasa Inggris untuk diproses oleh model.
- **Informasi Penyakit**: Menampilkan detail informasi penyakit yang diprediksi (melalui basis data `disease_info.json`).
- **Antarmuka Interaktif**: Desain antarmuka (UI) yang mudah digunakan dilengkapi fitur saran gejala (*symptom chips*).

## 📸 Screenshot Aplikasi

*(Silakan tambahkan file screenshot aplikasi Anda ke dalam folder proyek dan ubah nama file di bawah ini jika diperlukan)*

![Screenshot MedScan-AI](screenshot.png)

## 🚀 Instalasi & Persiapan

Pastikan Anda telah menginstal **Python 3.8+** di sistem Anda.

1. **Clone repository ini** (jika belum):
   ```bash
   git clone https://github.com/matthydmns028/MedScan-AI.git
   cd MedScan-AI
   ```

2. **Buat dan aktifkan Virtual Environment (Opsional namun disarankan)**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instal dependensi yang dibutuhkan**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Instruksi Penggunaan

1. **Jalankan Server Aplikasi**:
   Jalankan file `api.py` menggunakan Python untuk menyalakan backend dan frontend:
   ```bash
   python api.py
   ```
   Server Flask akan berjalan di port `5000`.

2. **Akses Melalui Browser**:
   Buka browser Anda dan kunjungi URL berikut:
   **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

3. **Cara Menggunakan**:
   - Masukkan gejala yang Anda alami di kolom teks. Anda bisa memasukkan lebih dari satu gejala dengan memisahkannya menggunakan tanda koma (contoh: *sakit kepala, demam tinggi, batuk*).
   - Sebagai alternatif, Anda dapat mengklik tombol saran gejala yang tersedia di layar.
   - Klik tombol **Prediksi** untuk melihat hasil analisis, tingkat persentase, dan informasi penyakit terkait.

## 🛠 Teknologi yang Digunakan
- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML, CSS, JavaScript (Vanilla)

## ⚠️ Disclaimer Medis
Aplikasi ini ditujukan murni untuk tujuan **edukasi dan informasi awal**. Hasil prediksi **bukanlah diagnosis medis resmi** dan tidak dapat menggantikan peran tenaga medis profesional. Selalu konsultasikan kondisi kesehatan Anda kepada dokter atau fasilitas kesehatan terdekat.
