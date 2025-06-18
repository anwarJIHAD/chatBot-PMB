import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
app = Flask(__name__)

# --- Load model dan komponen ---
model = tf.keras.models.load_model("model/intent_model.h5")

with open("model/tokenizer.json") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    
with open("model/label_encoder.json") as f:
    classes = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

with open("model/responses.json") as f:
    responses_dict = json.load(f)
    fallback_response = responses_dict.get("fallback", "Maaf, saya belum memahami pertanyaan Anda.")

with open("model/pattern_response_map.json", encoding="utf-8") as f:
    pattern_map = json.load(f)

max_len = model.input_shape[1]

# --- Precompute TF-IDF vectorizer dan matriks untuk setiap intent ---
vectorizers = {}
tfidf_matrices = {}

for intent, items in pattern_map.items():
    patterns = [item["pattern"] for item in items]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(patterns)
    vectorizers[intent] = vectorizer
    tfidf_matrices[intent] = tfidf_matrix

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Mohon masukkan pesan."})

    # Tokenisasi + padding
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    # Prediksi intent
    predictions = model.predict(padded)
    intent_index = np.argmax(predictions)
    confidence = predictions[0][intent_index]
    intent = label_encoder.inverse_transform([intent_index])[0]

    # Jika confidence rendah atau intent tidak dikenali
    if confidence < 0.6 or intent not in pattern_map:
        return jsonify({"response": fallback_response})

    # Cosine similarity menggunakan precompute vectorizer dan matrix
    vectorizer = vectorizers[intent]
    tfidf_matrix = tfidf_matrices[intent]

    message_vec = vectorizer.transform([message])
    sim = cosine_similarity(message_vec, tfidf_matrix)[0]
    best_match_index = np.argmax(sim)

    response = pattern_map[intent][best_match_index]["response"]

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

