import streamlit as st
import joblib

# Set konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Penyakit via NLP",
    page_icon="🩺",
    layout="centered"
)

# Load model (di-cache agar tidak perlu load ulang setiap kali interaksi)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('symptoms_disease_nlp_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Header Aplikasi
st.title("🩺 Medical Symptom to Disease Predictor")
st.markdown("""
Aplikasi web ini menggunakan *Machine Learning* berbasis NLP untuk memprediksi kemungkinan penyakit berdasarkan teks keluhan atau gejala yang Anda alami.
""")

if model is None:
    st.error("❌ Model tidak ditemukan! Pastikan Anda sudah melatih model dan file `symptoms_disease_nlp_model.pkl` tersedia di folder ini.")
else:
    # Input dari user
    st.subheader("📝 Masukkan Gejala Anda")
    symptoms_input = st.text_area(
        "Ceritakan keluhan Anda dalam bahasa Inggris (pisahkan dengan koma jika banyak):",
        placeholder="Contoh: shortness of breath, depressive or psychotic symptoms, dizziness, chest pain...",
        height=150
    )

    # Tombol Prediksi
    if st.button("🔍 Prediksi Penyakit", use_container_width=True):
        if symptoms_input.strip() == "":
            st.warning("⚠️ Mohon masukkan gejala Anda terlebih dahulu.")
        else:
            with st.spinner("Menelusuri database medis..."):
                # Melakukan prediksi
                prediction = model.predict([symptoms_input])[0]
                
                # Menampilkan hasil
                st.success("✅ **Hasil Prediksi Berhasil!**")
                st.markdown(f"<h2 style='text-align: center; color: #ff4b4b;'>{prediction.upper()}</h2>", unsafe_allow_html=True)
                
                # Disclaimer
                st.info("⚠️ **Disclaimer:** Aplikasi ini dibuat menggunakan Machine Learning untuk tujuan edukasi. Harap selalu konsultasikan dengan dokter profesional untuk diagnosis yang 100% akurat.")
