import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import save_model

# --- 1. Load dataset dari Excel ---
df = pd.read_excel("data/faq_dataset_final.xlsx")

# Validasi kolom
required_cols = {"intent", "pattern", "response"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError("Excel harus memiliki kolom: intent, pattern, response")

patterns = df["pattern"].values
intents = df["intent"].values
responses = df.drop_duplicates("intent").set_index("intent")["response"].to_dict()

# --- 2. Encode label intent ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(intents)
num_classes = len(set(labels))

# --- 3. Tokenisasi teks ---
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

# --- 4. Bangun model ---
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- 5. Training ---
model.fit(padded_sequences, labels, epochs=300)

# --- 6. Simpan model dan tokenizer ---
os.makedirs("model", exist_ok=True)
model.save("model/intent_model.h5")

with open("model/tokenizer.json", "w") as f:
    json.dump(tokenizer.to_json(), f)

with open("model/label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

# --- 7. Simpan respons intent ---
with open("model/responses.json", "w") as f:
    json.dump(responses, f)

print("✅ Training selesai dan model disimpan!")
