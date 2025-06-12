# api_server.py

import os
import re
import pickle
import joblib
import nltk
import numpy as np

import tensorflow as tf
import keras # Pastikan Keras diimpor untuk Keras 3+
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, jsonify
from flask_cors import CORS # Untuk menangani CORS

# --- KONFIGURASI DAN PEMUATAN NLTK/SASTRAWI (untuk preprocessing teks) ---
# Fungsi ini akan mencoba mengunduh data NLTK jika belum ada
def initialize_nlp_utils():
    global stop_words_indonesian, stemmer, word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Mengunduh punkt...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Mengunduh stopwords...")
        nltk.download('stopwords')

    from nltk.corpus import stopwords
    stop_words_indonesian = stopwords.words('indonesian')

    from nltk import word_tokenize
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    print("Utilitas NLP (stopwords, punkt, stemmer) siap.")

# Panggil inisialisasi utilitas NLP di awal
initialize_nlp_utils()

def preprocess_text_vak(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    print("Menggunakan word_tokenize dari:", word_tokenize.__module__)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_indonesian]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)
# -------------------------------------------------------------------------

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk semua rute. Sesuaikan untuk produksi jika perlu.

# --- DEFINISIKAN PATH DAN PARAMETER GLOBAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Untuk Model NLP
NLP_MODEL_DIR_NAME = "model_vak_nlp_tf_savedmodel"
NLP_MODEL_DIR = os.path.join(BASE_DIR, NLP_MODEL_DIR_NAME)
NLP_TOKENIZER_PATH = os.path.join(BASE_DIR, "keras_tokenizer_vak_nlp.pkl")
NLP_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_vak_nlp.pkl")
NLP_MAX_SEQUENCE_LENGTH = 30 # Verifikasi dari output model.export() -> TensorSpec shape
NLP_INPUT_DTYPE = tf.float32 # Verifikasi dari output model.export() -> TensorSpec dtype

# Untuk Model Kuesioner
QUIZ_MODEL_DIR_NAME = "model_kuesioner_vak_tf_savedmodel"
QUIZ_MODEL_DIR = os.path.join(BASE_DIR, QUIZ_MODEL_DIR_NAME)
QUIZ_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_kuesioner_vak.pkl")
QUIZ_NUM_FEATURES = 12 # Jumlah pertanyaan/fitur input Anda
QUIZ_INPUT_DTYPE = tf.float32 # Verifikasi dari output model_pilgan.export() -> TensorSpec dtype

# --- INISIALISASI VARIABEL MODEL GLOBAL ---
loaded_model_nlp_api = None
loaded_tokenizer_nlp_api = None
loaded_label_encoder_nlp_api = None
reverse_label_mapping_nlp_api = None

loaded_model_quiz_api = None
loaded_label_encoder_quiz_api = None
reverse_label_mapping_quiz_api = None

# --- FUNGSI UNTUK MEMUAT SUMBER DAYA MODEL ---
def load_nlp_resources():
    global loaded_model_nlp_api, loaded_tokenizer_nlp_api, loaded_label_encoder_nlp_api, reverse_label_mapping_nlp_api
    print(f"Memulai pemuatan sumber daya NLP...")
    try:
        print(f"Mencoba memuat model NLP dari: {NLP_MODEL_DIR}")
        if not os.path.exists(NLP_MODEL_DIR):
            raise FileNotFoundError(f"Direktori model NLP tidak ditemukan: {NLP_MODEL_DIR}")
        
        tfsm_layer_nlp = keras.layers.TFSMLayer(NLP_MODEL_DIR, call_endpoint='serve') # Asumsi endpoint 'serve'
        inputs_nlp = keras.Input(shape=(NLP_MAX_SEQUENCE_LENGTH,), dtype=NLP_INPUT_DTYPE)
        outputs_nlp = tfsm_layer_nlp(inputs_nlp)
        loaded_model_nlp_api = keras.Model(inputs_nlp, outputs_nlp)
        print("Model NLP (TFSMLayer wrapper) berhasil dimuat.")

        with open(NLP_TOKENIZER_PATH, "rb") as f:
            loaded_tokenizer_nlp_api = pickle.load(f)
        print("Keras Tokenizer NLP berhasil dimuat.")

        loaded_label_encoder_nlp_api = joblib.load(NLP_LABEL_ENCODER_PATH)
        reverse_label_mapping_nlp_api = {i: label for i, label in enumerate(loaded_label_encoder_nlp_api.classes_)}
        print(f"Label Encoder NLP berhasil dimuat. Mapping: {reverse_label_mapping_nlp_api}")
        print("Sumber daya NLP berhasil dimuat sepenuhnya.")

    except Exception as e:
        print(f"ERROR saat memuat sumber daya NLP untuk API: {e}")
        loaded_model_nlp_api = None # Set ke None jika gagal agar endpoint tahu

def load_quiz_resources():
    global loaded_model_quiz_api, loaded_label_encoder_quiz_api, reverse_label_mapping_quiz_api
    print(f"Memulai pemuatan sumber daya Kuesioner...")
    try:
        print(f"Mencoba memuat model Kuesioner dari: {QUIZ_MODEL_DIR}")
        if not os.path.exists(QUIZ_MODEL_DIR):
            raise FileNotFoundError(f"Direktori model Kuesioner tidak ditemukan: {QUIZ_MODEL_DIR}")

        # VERIFIKASI call_endpoint dan dtype input dari output model_pilgan.export()
        tfsm_layer_quiz = keras.layers.TFSMLayer(QUIZ_MODEL_DIR, call_endpoint='serve') # Asumsi endpoint 'serve'
        inputs_quiz = keras.Input(shape=(QUIZ_NUM_FEATURES,), dtype=QUIZ_INPUT_DTYPE)
        outputs_quiz = tfsm_layer_quiz(inputs_quiz)
        loaded_model_quiz_api = keras.Model(inputs_quiz, outputs_quiz)
        print("Model Kuesioner (TFSMLayer wrapper) berhasil dimuat.")

        loaded_label_encoder_quiz_api = joblib.load(QUIZ_LABEL_ENCODER_PATH)
        reverse_label_mapping_quiz_api = {i: label for i, label in enumerate(loaded_label_encoder_quiz_api.classes_)}
        print(f"Label Encoder Kuesioner berhasil dimuat. Mapping: {reverse_label_mapping_quiz_api}")
        print("Sumber daya Kuesioner berhasil dimuat sepenuhnya.")

    except Exception as e:
        print(f"ERROR saat memuat sumber daya Kuesioner untuk API: {e}")
        loaded_model_quiz_api = None # Set ke None jika gagal

# Panggil fungsi pemuatan saat aplikasi Flask dimulai (di luar konteks permintaan)
print("Memuat model dan artefak saat aplikasi dimulai...")
load_nlp_resources()
load_quiz_resources()
print("Pemuatan awal model dan artefak selesai.")


# --- ENDPOINTS API ---
@app.route('/predict-nlp', methods=['POST'])
def predict_nlp_endpoint(): # Ubah nama fungsi agar unik
    if not all([loaded_model_nlp_api, loaded_tokenizer_nlp_api, loaded_label_encoder_nlp_api, reverse_label_mapping_nlp_api]):
        return jsonify({'error': 'Model NLP atau artefak pendukung tidak berhasil dimuat di server'}), 500

    try:
        data_input = request.get_json(force=True)
        user_story = data_input.get('narasi_pengguna')

        if not user_story:
            return jsonify({'error': 'Field "narasi_pengguna" tidak ditemukan atau kosong'}), 400

        processed_story = preprocess_text_vak(user_story)
        sequences = loaded_tokenizer_nlp_api.texts_to_sequences([processed_story])
        padded_sequences = pad_sequences(sequences, maxlen=NLP_MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        input_tensor_nlp = padded_sequences.astype(NLP_INPUT_DTYPE.as_numpy_dtype() if hasattr(NLP_INPUT_DTYPE, 'as_numpy_dtype') else np.float32)
        
        prediction_probabilities = loaded_model_nlp_api.predict(input_tensor_nlp)
        predicted_class_encoded = np.argmax(prediction_probabilities, axis=1)
        predicted_style_label = loaded_label_encoder_nlp_api.inverse_transform(predicted_class_encoded)[0]

        return jsonify({
            'gaya_belajar_prediksi': predicted_style_label,
            'probabilitas': prediction_probabilities[0].tolist(),
            'narasi_asli': user_story,
            'teks_setelah_preprocessing': processed_story
        })
    except Exception as e:
        print(f"Error pada endpoint /predict-nlp: {e}")
        # traceback.print_exc() # Untuk debugging lebih detail di log server
        return jsonify({'error': f'Terjadi kesalahan pada server saat prediksi NLP: {str(e)}'}), 500

@app.route('/predict-quiz', methods=['POST'])
def predict_quiz_endpoint(): # Ubah nama fungsi agar unik
    if not all([loaded_model_quiz_api, loaded_label_encoder_quiz_api, reverse_label_mapping_quiz_api]):
        return jsonify({'error': 'Model Kuesioner atau artefak pendukung tidak berhasil dimuat di server'}), 500

    try:
        data_input = request.get_json(force=True)
        quiz_answers = data_input.get('quiz_answers') # Ekspektasi: array angka, misal [3,1,4,2,...]

        if not quiz_answers or not isinstance(quiz_answers, list) or len(quiz_answers) != QUIZ_NUM_FEATURES:
            return jsonify({'error': f'Field "quiz_answers" harus berupa array dengan {QUIZ_NUM_FEATURES} angka'}), 400

        # Konversi ke NumPy array dan pastikan dtype (sesuai QUIZ_INPUT_DTYPE)
        input_features_quiz = np.array([quiz_answers], dtype=QUIZ_INPUT_DTYPE.as_numpy_dtype() if hasattr(QUIZ_INPUT_DTYPE, 'as_numpy_dtype') else np.float32)
        
        prediction_probabilities = loaded_model_quiz_api.predict(input_features_quiz)
        predicted_class_encoded = np.argmax(prediction_probabilities, axis=1)
        predicted_style_label = loaded_label_encoder_quiz_api.inverse_transform(predicted_class_encoded)[0]

        return jsonify({
            'gaya_belajar_prediksi': predicted_style_label,
            'probabilitas': prediction_probabilities[0].tolist(),
            'jawaban_input': quiz_answers
        })
    except Exception as e:
        print(f"Error pada endpoint /predict-quiz: {e}")
        # traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan pada server saat prediksi Kuesioner: {str(e)}'}), 500
        
# Endpoint sederhana untuk cek status
@app.route('/status', methods=['GET'])
def status():
    nlp_status = "OK" if all([loaded_model_nlp_api, loaded_tokenizer_nlp_api, loaded_label_encoder_nlp_api]) else "ERROR"
    quiz_status = "OK" if all([loaded_model_quiz_api, loaded_label_encoder_quiz_api]) else "ERROR"
    return jsonify({
        "message": "StudyFinder ML API is running!",
        "nlp_model_status": nlp_status,
        "quiz_model_status": quiz_status
    })

if __name__ == '__main__':
    # Port untuk Railway (dari variabel lingkungan) atau default ke 5000 untuk lokal
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False untuk lingkungan produksi seperti Railway
    # Untuk Gunicorn, biasanya host dan port diatur oleh Gunicorn, tapi ini tidak masalah untuk lokal.
    app.run(host='0.0.0.0', port=port, debug=False)