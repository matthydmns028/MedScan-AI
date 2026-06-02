import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def main():
    print("=" * 60)
    print("  MedScan AI — Model Training Pipeline v2.0")
    print("=" * 60)

    print("\n[1/6] Memuat dataset final_symptoms_to_disease.csv...")
    try:
        df = pd.read_csv('final_symptoms_to_disease.csv')
    except FileNotFoundError:
        print("Error: File 'final_symptoms_to_disease.csv' tidak ditemukan di folder saat ini.")
        return

    # Pastikan tidak ada nilai kosong
    df = df.dropna(subset=['diseases', 'symptom_text'])

    # Hapus duplikat
    before_count = len(df)
    df = df.drop_duplicates()
    after_count = len(df)
    dups_removed = before_count - after_count
    print(f"   Total data mentah  : {before_count}")
    print(f"   Duplikat dihapus   : {dups_removed}")
    print(f"   Total data bersih  : {after_count}")
    print(f"   Jumlah penyakit    : {df['diseases'].nunique()}")

    X = df['symptom_text']
    y = df['diseases']

    print("\n[2/6] Membagi data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Data training : {len(X_train)}")
    print(f"   Data testing  : {len(X_test)}")

    # === Enhanced TF-IDF + CalibratedClassifierCV ===
    print("\n[3/6] Membuat pipeline NLP yang ditingkatkan...")
    print("   - TfidfVectorizer: sublinear_tf=True, ngram_range=(1,3), min_df=2, max_df=0.95")
    print("   - LinearSVC -> CalibratedClassifierCV (Platt Scaling)")
    print("   - Ini akan menghasilkan probabilitas nyata, bukan skor softmax hack")

    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),       # Uni-, bi-, dan tri-gram
        max_df=0.95,              # Hapus kata yang terlalu umum
        min_df=2,                 # Hapus kata yang hanya muncul sekali
        sublinear_tf=True,        # Logarithmic TF scaling
        max_features=50000        # Batasi fitur untuk efisiensi
    )

    base_svm = LinearSVC(
        random_state=42,
        max_iter=2000,            # Lebih banyak iterasi untuk konvergensi
        C=1.0                     # Regularization parameter
    )

    # CalibratedClassifierCV membungkus SVM untuk menghasilkan predict_proba()
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')

    model = make_pipeline(tfidf, calibrated_svm)

    print("\n[4/6] Melatih model (ini mungkin memakan waktu beberapa menit)...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"   Waktu pelatihan: {train_time:.1f} detik")

    # Evaluasi model
    print("\n[5/6] Mengevaluasi model pada data testing...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n   [OK] Akurasi Model: {acc * 100:.2f}%")

    # Tampilkan ringkasan classification report
    print("\n   Laporan Klasifikasi (macro avg):")
    report = classification_report(y_test, y_pred, output_dict=True)
    macro = report['macro avg']
    print(f"   - Precision : {macro['precision'] * 100:.2f}%")
    print(f"   - Recall    : {macro['recall'] * 100:.2f}%")
    print(f"   - F1-Score  : {macro['f1-score'] * 100:.2f}%")

    # Menyimpan model
    model_filename = 'symptoms_disease_nlp_model.pkl'
    print(f"\n[6/6] Menyimpan model ke '{model_filename}'...")
    joblib.dump(model, model_filename)
    print(f"   [OK] Model berhasil disimpan!")

    # Contoh prediksi dengan probabilitas
    print("\n" + "=" * 60)
    print("  Contoh Prediksi dengan Probabilitas")
    print("=" * 60)
    sample_symptoms = [
        "shortness of breath, depressive or psychotic symptoms, dizziness",
        "throat pain, fever, cough, headache",
        "chest pain, sweating, nausea, shortness of breath"
    ]

    for symptom_text in sample_symptoms:
        prediction = model.predict([symptom_text])[0]
        probas = model.predict_proba([symptom_text])[0]
        top_indices = probas.argsort()[::-1][:3]

        print(f"\n   Gejala: {symptom_text}")
        print(f"   Top 3 Prediksi:")
        for rank, idx in enumerate(top_indices, 1):
            disease = model.classes_[idx]
            prob = probas[idx] * 100
            print(f"     {rank}. {disease} — {prob:.2f}%")

    print("\n" + "=" * 60)
    print("  Training selesai! Model siap digunakan.")
    print("=" * 60)

if __name__ == "__main__":
    main()
