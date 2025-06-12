import os
import re
import pickle
import joblib
import numpy as np
import tensorflow as tf
import keras
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Konfigurasi dan utilitas NLP
def initialize_nlp_utils():
    global stop_words_indonesian, stemmer
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
    except LookupError:
        import nltk
        nltk.download('stopwords')

    stop_words_indonesian = stopwords.words('indonesian')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    print("Utilitas NLP (stopwords, stemmer) siap.")

initialize_nlp_utils()

def preprocess_text_vak(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()

    # Gunakan regex-based tokenizer untuk menghindari ketergantungan NLTK punkt
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [word for word in tokens if word not in stop_words_indonesian]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Flask setup
app = Flask(__name__)
CORS(app)

# Path dan parameter model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_MODEL_DIR = os.path.join(BASE_DIR, "model_vak_nlp_tf_savedmodel")
NLP_TOKENIZER_PATH = os.path.join(BASE_DIR, "keras_tokenizer_vak_nlp.pkl")
NLP_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_vak_nlp.pkl")
NLP_MAX_SEQUENCE_LENGTH = 30
NLP_INPUT_DTYPE = tf.float32

QUIZ_MODEL_DIR = os.path.join(BASE_DIR, "model_kuesioner_vak_tf_savedmodel")
QUIZ_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_kuesioner_vak.pkl")
QUIZ_NUM_FEATURES = 12
QUIZ_INPUT_DTYPE = tf.float32

# Variabel model global
loaded_model_nlp_api = None
loaded_tokenizer_nlp_api = None
loaded_label_encoder_nlp_api = None
reverse_label_mapping_nlp_api = None

loaded_model_quiz_api = None
loaded_label_encoder_quiz_api = None
reverse_label_mapping_quiz_api = None

# Fungsi pemuatan model
def load_nlp_resources():
    global loaded_model_nlp_api, loaded_tokenizer_nlp_api, loaded_label_encoder_nlp_api, reverse_label_mapping_nlp_api
    print("Memulai pemuatan sumber daya NLP...")
    try:
        tfsm_layer_nlp = keras.layers.TFSMLayer(NLP_MODEL_DIR, call_endpoint='serve')
        inputs_nlp = keras.Input(shape=(NLP_MAX_SEQUENCE_LENGTH,), dtype=NLP_INPUT_DTYPE)
        outputs_nlp = tfsm_layer_nlp(inputs_nlp)
        loaded_model_nlp_api = keras.Model(inputs_nlp, outputs_nlp)

        with open(NLP_TOKENIZER_PATH, "rb") as f:
            loaded_tokenizer_nlp_api = pickle.load(f)

        loaded_label_encoder_nlp_api = joblib.load(NLP_LABEL_ENCODER_PATH)
        reverse_label_mapping_nlp_api = {i: label for i, label in enumerate(loaded_label_encoder_nlp_api.classes_)}
        print("Sumber daya NLP berhasil dimuat.")
    except Exception as e:
        print(f"ERROR NLP: {e}")
        loaded_model_nlp_api = None

def load_quiz_resources():
    global loaded_model_quiz_api, loaded_label_encoder_quiz_api, reverse_label_mapping_quiz_api
    print("Memulai pemuatan sumber daya Kuesioner...")
    try:
        tfsm_layer_quiz = keras.layers.TFSMLayer(QUIZ_MODEL_DIR, call_endpoint='serve')
        inputs_quiz = keras.Input(shape=(QUIZ_NUM_FEATURES,), dtype=QUIZ_INPUT_DTYPE)
        outputs_quiz = tfsm_layer_quiz(inputs_quiz)
        loaded_model_quiz_api = keras.Model(inputs_quiz, outputs_quiz)

        loaded_label_encoder_quiz_api = joblib.load(QUIZ_LABEL_ENCODER_PATH)
        reverse_label_mapping_quiz_api = {i: label for i, label in enumerate(loaded_label_encoder_quiz_api.classes_)}
        print("Sumber daya Kuesioner berhasil dimuat.")
    except Exception as e:
        print(f"ERROR Quiz: {e}")
        loaded_model_quiz_api = None

# Panggil saat start
load_nlp_resources()
load_quiz_resources()

# Endpoints
@app.route('/predict-nlp', methods=['POST'])
def predict_nlp_endpoint():
    if not all([loaded_model_nlp_api, loaded_tokenizer_nlp_api, loaded_label_encoder_nlp_api]):
        return jsonify({'error': 'Model NLP tidak berhasil dimuat'}), 500
    try:
        data = request.get_json(force=True)
        user_story = data.get('narasi_pengguna')
        if not user_story:
            return jsonify({'error': 'Field narasi_pengguna tidak ditemukan'}), 400

        processed = preprocess_text_vak(user_story)
        seq = loaded_tokenizer_nlp_api.texts_to_sequences([processed])
        pad_seq = pad_sequences(seq, maxlen=NLP_MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        input_tensor = pad_seq.astype(NLP_INPUT_DTYPE.as_numpy_dtype() if hasattr(NLP_INPUT_DTYPE, 'as_numpy_dtype') else np.float32)

        pred = loaded_model_nlp_api.predict(input_tensor)
        pred_class = np.argmax(pred, axis=1)
        label = loaded_label_encoder_nlp_api.inverse_transform(pred_class)[0]

        return jsonify({
            'gaya_belajar_prediksi': label,
            'probabilitas': pred[0].tolist(),
            'narasi_asli': user_story,
            'teks_setelah_preprocessing': processed
        })
    except Exception as e:
        print(f"Error NLP: {e}")
        return jsonify({'error': f'Terjadi kesalahan pada server saat prediksi NLP: {str(e)}'}), 500

@app.route('/predict-quiz', methods=['POST'])
def predict_quiz_endpoint():
    if not all([loaded_model_quiz_api, loaded_label_encoder_quiz_api]):
        return jsonify({'error': 'Model Kuesioner tidak berhasil dimuat'}), 500
    try:
        data = request.get_json(force=True)
        quiz_answers = data.get('quiz_answers')
        if not quiz_answers or not isinstance(quiz_answers, list) or len(quiz_answers) != QUIZ_NUM_FEATURES:
            return jsonify({'error': f'"quiz_answers" harus array dengan {QUIZ_NUM_FEATURES} elemen'}), 400

        input_array = np.array([quiz_answers], dtype=QUIZ_INPUT_DTYPE.as_numpy_dtype() if hasattr(QUIZ_INPUT_DTYPE, 'as_numpy_dtype') else np.float32)
        pred = loaded_model_quiz_api.predict(input_array)
        pred_class = np.argmax(pred, axis=1)
        label = loaded_label_encoder_quiz_api.inverse_transform(pred_class)[0]

        return jsonify({
            'gaya_belajar_prediksi': label,
            'probabilitas': pred[0].tolist(),
            'jawaban_input': quiz_answers
        })
    except Exception as e:
        print(f"Error Quiz: {e}")
        return jsonify({'error': f'Terjadi kesalahan pada server saat prediksi Kuesioner: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "message": "StudyFinder ML API is running!",
        "nlp_model_status": "OK" if loaded_model_nlp_api else "ERROR",
        "quiz_model_status": "OK" if loaded_model_quiz_api else "ERROR"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
