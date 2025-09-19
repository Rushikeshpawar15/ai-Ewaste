from flask import Flask, request, jsonify, render_template
import joblib
import json
import re
import warnings
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------
# 1️⃣ Load trained models
# ------------------------
# NLP model
text_model = joblib.load("ewaste_model.pkl")
print("✅ NLP model loaded!")

# Image model
image_model = load_model("ewaste_image_model_final.h5")
print("✅ Image model loaded!")

# E-waste metadata
with open("ewaste_data.json", "r") as f:
    ewaste_info = json.load(f)
print("✅ E-waste metadata loaded!")

# ------------------------
# Label Mapping (old → new)
# ------------------------
label_mapping = {
    "Mobile": "mobile phone",
    "Television": "television",
    "Keyboard": "keyboard",
    "Microwave": "microwave oven",
    "mouse": "mouse",
    "Player": "dvd player",
    "Printer": "printer",
    "Washing Machine": "washing machine",
    "pcb": "pcb"
}

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Optional: clean text input
def clean_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z0-9 ]', '', query)
    return query

# ------------------------
# Home route
# ------------------------
@app.route('/')
def home():
    return render_template("index.html")

# ------------------------
# Predict route (text & image)
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # 1️⃣ Check for image file
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        IMG_SIZE = (224,224)
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred_probs = image_model.predict(img_array)
        class_index = np.argmax(pred_probs)

        # Ensure order matches training set
        class_names = list(sorted(os.listdir("E-Waste Dataset/train")))
        raw_category = class_names[class_index]

        # Apply label mapping
        category = label_mapping.get(raw_category, raw_category)

        # Get metadata
        info = ewaste_info.get(category, {})
        result = {
            "type": "image",
            "predicted_category": category,
            "disposal": info.get("disposal", "No info available"),
            "metals": info.get("metals", []),
            "hazards": info.get("hazards", [])
        }
        return jsonify(result)

    # 2️⃣ Text input
    query = request.form.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    cleaned = clean_query(query)
    raw_category = text_model.predict([cleaned])[0]

    # Apply label mapping
    category = label_mapping.get(raw_category, raw_category)

    info = ewaste_info.get(category, {})
    result = {
        "type": "text",
        "query": query,
        "predicted_category": category,
        "disposal": info.get("disposal", "No info available"),
        "metals": info.get("metals", []),
        "hazards": info.get("hazards", [])
    }
    return jsonify(result)

# ------------------------
# Run app
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)