from flask import Flask, request, jsonify
from services.embeddings import EmbeddingService
from services.vector_db import VectorDB
from services.cnn_predict import CNNPredictor
from services.hf_ocr import HFOCR      # Nougat OCR
from services.ocr_service import OCRService   # Tesseract fallback
from flask_cors import CORS

import os

app = Flask(__name__)
CORS(app)

# --- Initialize Services ---
embedder = EmbeddingService()
db = VectorDB()

hf_ocr = HFOCR()        # ⚡ New HuggingFace Nougat OCR
ocr = OCRService()      # Fallback only
cnn = CNNPredictor(
    model_path="models/cnn_transfer.keras",
    class_json="models/class_names.json"
)


# --------------------------------------------------
# MODULE 1 — TEXT PRODUCT RECOMMENDATION
# --------------------------------------------------
@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.form.get('query', '')

    if not query:
        return jsonify({
            "products": [],
            "response": "Please provide a product query."
        })

    query_vec = embedder.embed_text([query])[0]
    results = db.query(query_vec, top_k=5)

    products = [{
        "id": m.id,
        "score": float(m.score),
        "product_name": m.metadata.get("product_name", ""),
        "price": m.metadata.get("price", "")
    } for m in results.matches]

    return jsonify({
        "products": products,
        "response": f"Here are the top {len(products)} recommendations for '{query}'."
    })


# --------------------------------------------------
# MODULE 2 — OCR (HF Nougat → Tesseract fallback)
# --------------------------------------------------
@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    file = request.files.get("image_data")

    if file is None:
        return jsonify({
            "products": [],
            "response": "No image uploaded.",
            "extracted_text": ""
        })

    # --- Try HuggingFace OCR First ---
    file.seek(0)
    extracted = hf_ocr.extract_text(file)

    # If failed, fallback to Tesseract
    if not extracted or len(extracted.strip()) < 2:
        file.seek(0)
        extracted = ocr.extract_text(file)

    if not extracted or extracted.strip() == "":
        return jsonify({
            "products": [],
            "response": "Could not extract readable text.",
            "extracted_text": ""
        })

    # Vector DB search
    vec = embedder.embed_text([extracted])[0]
    results = db.query(vec, top_k=5)

    products = [{
        "id": m.id,
        "score": float(m.score),
        "product_name": m.metadata.get("product_name", ""),
        "price": m.metadata.get("price", "")
    } for m in results.matches]

    return jsonify({
        "products": products,
        "response": f"OCR text '{extracted}' matched these products.",
        "extracted_text": extracted
    })


# --------------------------------------------------
# MODULE 3 — CNN IMAGE PRODUCT CLASSIFICATION
# --------------------------------------------------
@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    file = request.files.get("product_image")

    if file is None:
        return jsonify({
            "class": "",
            "products": [],
            "response": "No image uploaded."
        })

    temp_path = "tmp_upload.jpg"
    file.save(temp_path)

    prediction = cnn.predict(temp_path)
    predicted_class = prediction["class"]

    vec = embedder.embed_text([predicted_class])[0]
    results = db.query(vec, top_k=5)

    products = [{
        "id": m.id,
        "score": float(m.score),
        "product_name": m.metadata.get("product_name", ""),
        "price": m.metadata.get("price", "")
    } for m in results.matches]

    return jsonify({
        "class": predicted_class,
        "products": products,
        "response": f"Predicted class '{predicted_class}'. Here are similar products."
    })


# --------------------------------------------------
# START SERVER
# --------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
