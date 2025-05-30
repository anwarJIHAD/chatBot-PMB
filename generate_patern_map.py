import pandas as pd
import json
import os

# Baca file Excel
df = pd.read_excel("data/faq_dataset_final.xlsx")

# Buat dictionary { intent: [ {pattern, response}, ... ] }
intent_map = {}

for _, row in df.iterrows():
    intent = row["intent"]
    pattern = row["pattern"]
    response = row["response"]

    if intent not in intent_map:
        intent_map[intent] = []

    intent_map[intent].append({
        "pattern": pattern,
        "response": response
    })

# Simpan ke JSON
os.makedirs("model", exist_ok=True)
with open("model/pattern_response_map.json", "w", encoding="utf-8") as f:
    json.dump(intent_map, f, ensure_ascii=False, indent=2)

print("✅ File pattern_response_map.json berhasil dibuat!")
