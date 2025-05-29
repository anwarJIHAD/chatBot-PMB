from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- Load model, tokenizer, label encoder, and response map ---
model = tf.keras.models.load_model("model/intent_model.h5")

with open("model/tokenizer.json") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

with open("model/label_encoder.json") as f:
    classes = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

with open("model/responses.json") as f:
    responses = json.load(f)

# Tentukan panjang maksimal input
max_len = model.input_shape[1]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")

    # Tokenisasi dan padding
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")

    # Prediksi intent
    predictions = model.predict(padded)
    intent_index = np.argmax(predictions)
    intent = label_encoder.inverse_transform([intent_index])[0]

    # Ambil respon dari intent
    response = responses.get(intent, "Maaf, saya belum mengerti pertanyaan Anda.")

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
