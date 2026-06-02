from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import re
import logging
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Mendapatkan direktori tempat file api.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # Mengizinkan HTML/Frontend mengakses API ini

# Load model saat server mulai berjalan (menggunakan absolute path)
MODEL_PATH = os.path.join(BASE_DIR, 'symptoms_disease_nlp_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model ML berhasil dimuat dari: {MODEL_PATH}")
except FileNotFoundError:
    model = None
    logger.warning(f"Model tidak ditemukan di: {MODEL_PATH}")

# Load disease info knowledge base
DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.json')
disease_info_db = {}
try:
    with open(DISEASE_INFO_PATH, 'r', encoding='utf-8') as f:
        disease_info_db = json.load(f)
    logger.info(f"Disease info dimuat: {len(disease_info_db)} penyakit")
except FileNotFoundError:
    logger.warning(f"Disease info tidak ditemukan di: {DISEASE_INFO_PATH}")
except json.JSONDecodeError:
    logger.error(f"Disease info JSON tidak valid: {DISEASE_INFO_PATH}")


# ====================================
# Kamus terjemahan gejala ID -> EN
# ====================================
SYMPTOM_ID_TO_EN = {
    # Gejala umum
    "sakit kepala": "headache",
    "pusing": "dizziness",
    "demam": "fever",
    "demam tinggi": "high fever",
    "menggigil": "chills",
    "kelelahan": "fatigue",
    "lelah": "fatigue",
    "capek": "fatigue",
    "lemas": "weakness",
    "lemah": "weakness",
    "badan lemas": "weakness",
    "tidak nafsu makan": "loss of appetite",
    "nafsu makan menurun": "loss of appetite",
    "penurunan berat badan": "weight loss",
    "berat badan turun": "weight loss",
    "berat badan naik": "weight gain",
    "keringat berlebih": "excessive sweating",
    "keringat malam": "night sweats",
    "insomnia": "insomnia",
    "susah tidur": "insomnia",
    "sulit tidur": "insomnia",
    "mengantuk": "drowsiness",
    "pingsan": "fainting",
    "kejang": "seizures",
    "bengkak": "swelling",
    "pembengkakan": "swelling",
    "nyeri": "pain",
    "sakit": "pain",

    # Pernapasan
    "batuk": "cough",
    "batuk kering": "dry cough",
    "batuk berdahak": "productive cough",
    "batuk berdarah": "coughing blood",
    "sesak napas": "shortness of breath",
    "sesak nafas": "shortness of breath",
    "napas pendek": "shortness of breath",
    "nafas pendek": "shortness of breath",
    "sulit bernapas": "difficulty breathing",
    "sulit bernafas": "difficulty breathing",
    "pilek": "runny nose",
    "hidung tersumbat": "nasal congestion",
    "hidung mampet": "nasal congestion",
    "bersin": "sneezing",
    "mengi": "wheezing",
    "napas berbunyi": "wheezing",
    "nafas berbunyi": "wheezing",
    "sakit tenggorokan": "sore throat",
    "radang tenggorokan": "sore throat",
    "tenggorokan sakit": "sore throat",
    "nyeri tenggorokan": "throat pain",
    "suara serak": "hoarseness",
    "dahak": "phlegm",

    # Pencernaan
    "mual": "nausea",
    "muntah": "vomiting",
    "diare": "diarrhea",
    "mencret": "diarrhea",
    "sembelit": "constipation",
    "susah bab": "constipation",
    "sakit perut": "abdominal pain",
    "nyeri perut": "abdominal pain",
    "perut kembung": "bloating",
    "kembung": "bloating",
    "perut begah": "bloating",
    "maag": "gastritis",
    "asam lambung": "acid reflux",
    "heartburn": "heartburn",
    "nyeri ulu hati": "heartburn",
    "perut mulas": "stomach cramps",
    "buang air besar berdarah": "blood in stool",
    "bab berdarah": "blood in stool",
    "wasir": "hemorrhoids",
    "ambeien": "hemorrhoids",
    "sulit menelan": "difficulty swallowing",
    "begah": "indigestion",

    # Jantung & pembuluh darah
    "nyeri dada": "chest pain",
    "sakit dada": "chest pain",
    "dada sesak": "chest tightness",
    "jantung berdebar": "palpitations",
    "detak jantung cepat": "rapid heartbeat",
    "detak jantung tidak teratur": "irregular heartbeat",
    "tekanan darah tinggi": "high blood pressure",
    "tekanan darah rendah": "low blood pressure",
    "kaki bengkak": "swollen legs",

    # Kulit
    "ruam": "rash",
    "ruam kulit": "skin rash",
    "gatal": "itching",
    "gatal-gatal": "itching",
    "kulit kering": "dry skin",
    "kulit merah": "skin redness",
    "kulit mengelupas": "skin peeling",
    "jerawat": "acne",
    "bisul": "boil",
    "luka": "wound",
    "luka tidak sembuh": "non-healing wound",
    "memar": "bruising",
    "kulit kuning": "jaundice",
    "mata kuning": "jaundice",
    "biduran": "hives",
    "bentol": "hives",
    "kulit gatal": "skin itching",
    "eksim": "eczema",

    # Otot & sendi
    "nyeri sendi": "joint pain",
    "sakit sendi": "joint pain",
    "nyeri otot": "muscle pain",
    "sakit otot": "muscle pain",
    "pegal": "muscle ache",
    "kaku sendi": "joint stiffness",
    "nyeri punggung": "back pain",
    "sakit punggung": "back pain",
    "nyeri pinggang": "lower back pain",
    "sakit pinggang": "lower back pain",
    "nyeri leher": "neck pain",
    "sakit leher": "neck pain",
    "nyeri bahu": "shoulder pain",
    "kram otot": "muscle cramps",
    "kram": "cramps",
    "kesemutan": "tingling",
    "mati rasa": "numbness",
    "kebas": "numbness",

    # Saluran kemih
    "sering buang air kecil": "frequent urination",
    "sering kencing": "frequent urination",
    "nyeri saat kencing": "painful urination",
    "sakit saat kencing": "painful urination",
    "kencing berdarah": "blood in urine",
    "air kencing keruh": "cloudy urine",
    "sulit buang air kecil": "difficulty urinating",
    "ngompol": "urinary incontinence",

    # Mata
    "mata merah": "red eyes",
    "mata gatal": "itchy eyes",
    "mata berair": "watery eyes",
    "penglihatan kabur": "blurred vision",
    "pandangan kabur": "blurred vision",
    "mata kering": "dry eyes",
    "nyeri mata": "eye pain",
    "sakit mata": "eye pain",
    "sensitif cahaya": "light sensitivity",

    # Telinga
    "sakit telinga": "ear pain",
    "nyeri telinga": "ear pain",
    "telinga berdenging": "ringing in ears",
    "pendengaran menurun": "hearing loss",
    "telinga berair": "ear discharge",
    "telinga gatal": "itchy ear",

    # Mental & saraf
    "cemas": "anxiety",
    "gelisah": "anxiety and nervousness",
    "khawatir": "anxiety",
    "depresi": "depression",
    "sedih": "depression",
    "stres": "stress",
    "gangguan tidur": "sleep disturbance",
    "halusinasi": "hallucinations",
    "bingung": "confusion",
    "linglung": "confusion",
    "gangguan konsentrasi": "difficulty concentrating",
    "mudah marah": "irritability",
    "serangan panik": "panic attack",
    "tremor": "tremor",
    "gemetar": "tremor",

    # Lainnya
    "mimisan": "nosebleed",
    "hidung berdarah": "nosebleed",
    "sariawan": "mouth ulcer",
    "gusi berdarah": "bleeding gums",
    "sakit gigi": "toothache",
    "nyeri gigi": "toothache",
    "kelenjar getah bening bengkak": "swollen lymph nodes",
    "haus berlebihan": "excessive thirst",
    "sering haus": "increased thirst",
    "lapar berlebihan": "extreme hunger",
    "mulut kering": "dry mouth",
    "rambut rontok": "hair loss",
    "alergi": "allergy",
    "berkeringat dingin": "cold sweats",
}

# ====================================
# Common symptom chips untuk UI
# ====================================
COMMON_SYMPTOMS = [
    {"id": "headache", "label_id": "Sakit Kepala", "label_en": "Headache"},
    {"id": "fever", "label_id": "Demam", "label_en": "Fever"},
    {"id": "cough", "label_id": "Batuk", "label_en": "Cough"},
    {"id": "fatigue", "label_id": "Kelelahan", "label_en": "Fatigue"},
    {"id": "nausea", "label_id": "Mual", "label_en": "Nausea"},
    {"id": "dizziness", "label_id": "Pusing", "label_en": "Dizziness"},
    {"id": "shortness of breath", "label_id": "Sesak Napas", "label_en": "Shortness of Breath"},
    {"id": "chest pain", "label_id": "Nyeri Dada", "label_en": "Chest Pain"},
    {"id": "sore throat", "label_id": "Sakit Tenggorokan", "label_en": "Sore Throat"},
    {"id": "diarrhea", "label_id": "Diare", "label_en": "Diarrhea"},
    {"id": "vomiting", "label_id": "Muntah", "label_en": "Vomiting"},
    {"id": "abdominal pain", "label_id": "Sakit Perut", "label_en": "Abdominal Pain"},
    {"id": "back pain", "label_id": "Nyeri Punggung", "label_en": "Back Pain"},
    {"id": "joint pain", "label_id": "Nyeri Sendi", "label_en": "Joint Pain"},
    {"id": "skin rash", "label_id": "Ruam Kulit", "label_en": "Skin Rash"},
    {"id": "insomnia", "label_id": "Insomnia", "label_en": "Insomnia"},
    {"id": "anxiety", "label_id": "Cemas", "label_en": "Anxiety"},
    {"id": "blurred vision", "label_id": "Penglihatan Kabur", "label_en": "Blurred Vision"},
    {"id": "swelling", "label_id": "Bengkak", "label_en": "Swelling"},
    {"id": "chills", "label_id": "Menggigil", "label_en": "Chills"},
]


def translate_symptoms(text):
    """
    Menerjemahkan gejala dari bahasa Indonesia ke bahasa Inggris.
    Mendukung input campuran (sebagian ID, sebagian EN).
    Mengembalikan tuple: (teks_terjemahan, daftar_terjemahan).
    """
    text_lower = text.lower().strip()

    # Pisahkan berdasarkan koma
    parts = [p.strip() for p in text_lower.split(',') if p.strip()]

    translated_parts = []
    translations_log = []  # Untuk menampilkan ke user

    for part in parts:
        # Coba cocokkan dengan kamus (dari frasa terpanjang dulu)
        matched = False
        for id_term, en_term in sorted(SYMPTOM_ID_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
            if part == id_term or part.strip() == id_term:
                translated_parts.append(en_term)
                translations_log.append({'from': part, 'to': en_term})
                matched = True
                break

        if not matched:
            # Tidak ditemukan di kamus — asumsikan sudah bahasa Inggris
            translated_parts.append(part)

    return ', '.join(translated_parts), translations_log


def get_confidence_level(probability):
    """Mengembalikan label tingkat kepercayaan berdasarkan probabilitas."""
    if probability >= 70:
        return {"level": "high", "label": "Tinggi", "color": "#00e6a7"}
    elif probability >= 40:
        return {"level": "medium", "label": "Sedang", "color": "#f59e0b"}
    elif probability >= 20:
        return {"level": "low", "label": "Rendah", "color": "#f87171"}
    else:
        return {"level": "very_low", "label": "Sangat Rendah", "color": "#64748b"}


# Route utama — melayani halaman HTML
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

# Melayani file CSS dan asset statis lainnya
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


# ====================================
# Health Check Endpoint
# ====================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'disease_info_count': len(disease_info_db),
        'translation_entries': len(SYMPTOM_ID_TO_EN),
        'timestamp': datetime.now().isoformat()
    })


# ====================================
# Symptom Suggestions Endpoint
# ====================================
@app.route('/symptom-suggestions', methods=['GET'])
def symptom_suggestions():
    return jsonify({'symptoms': COMMON_SYMPTOMS})


# ====================================
# Disease Info Endpoint
# ====================================
@app.route('/disease-info/<path:disease_name>', methods=['GET'])
def disease_info(disease_name):
    name_lower = disease_name.lower().strip()
    info = disease_info_db.get(name_lower)
    if info:
        return jsonify({
            'found': True,
            'disease': name_lower,
            **info
        })
    else:
        return jsonify({
            'found': False,
            'disease': name_lower,
            'message': 'Informasi detail belum tersedia untuk penyakit ini.'
        })


# ====================================
# Main Prediction Endpoint
# ====================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dilatih atau file model tidak ditemukan.'}), 500

    data = request.get_json()

    if not data or 'symptoms' not in data:
        return jsonify({'error': 'Input gejala tidak valid.'}), 400

    symptoms_text = data['symptoms']

    # Validasi panjang input
    if len(symptoms_text) > 500:
        return jsonify({'error': 'Input terlalu panjang. Maksimum 500 karakter.'}), 400

    if len(symptoms_text.strip()) < 3:
        return jsonify({'error': 'Input terlalu pendek. Masukkan minimal satu gejala.'}), 400

    top_n = data.get('top_n', 5)

    try:
        # Terjemahkan gejala dari ID ke EN (jika ada)
        translated_text, translations = translate_symptoms(symptoms_text)
        logger.info(f"Prediksi diminta — Input: '{symptoms_text}' → Translated: '{translated_text}'")

        # Gunakan predict_proba jika tersedia (CalibratedClassifierCV)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([translated_text])[0]
        else:
            # Fallback: gunakan decision_function + softmax
            decision_scores = model.decision_function([translated_text])[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()

        # Mendapatkan nama-nama kelas dari model
        classes = model.classes_

        # Mengurutkan berdasarkan probabilitas tertinggi
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        # Membuat daftar hasil prediksi
        predictions = []
        for idx in top_indices:
            disease_name = classes[idx]
            prob = round(float(probabilities[idx]) * 100, 2)
            confidence = get_confidence_level(prob)

            # Cek apakah info penyakit tersedia
            info = disease_info_db.get(disease_name.lower(), None)
            has_info = info is not None

            predictions.append({
                'disease': disease_name,
                'probability': prob,
                'confidence': confidence,
                'has_info': has_info
            })

        top_disease = predictions[0]['disease']
        logger.info(f"Hasil prediksi: {top_disease} ({predictions[0]['probability']}%)")

        return jsonify({
            'predictions': predictions,
            'top_prediction': top_disease,
            'translated_input': translated_text,
            'translations': translations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error saat prediksi: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Server akan berjalan pada port 5000 http://127.0.0.1:5000
    app.run(debug=True, port=5000)
