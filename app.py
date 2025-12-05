from flask import Flask, request, jsonify
from flask_cors import CORS

from services.embeddings import EmbeddingService
from services.vector_db import VectorDB
from services.ocr_service import EasyOCRService
from services.cnn_predict import CNNPredictor

import os

app = Flask(__name__)
CORS(app)

# -----------------------
# INITIALIZE SERVICES
# -----------------------
embedder = EmbeddingService()
db = VectorDB()

ocr = EasyOCRService()      # 🔥 main OCR (EasyOCR headless)

cnn = CNNPredictor(
    model_path="models/cnn_transfer.keras",
    class_json="models/class_names.json"
)

# --------------------------------------------------
# MODULE 1 — PRODUCT RECOMMENDATION (TEXT SEARCH)
# --------------------------------------------------
@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.form.get('query', '')

    if not query:
        return jsonify({
            "products": [],
            "response": "Please provide a product query."
        })

    vec = embedder.embed_text([query])[0]
    results = db.query(vec, top_k=5)

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
# MODULE 2 — OCR + Product Search
# --------------------------------------------------
@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    image_file = request.files.get("image_data")

    if image_file is None:
        return jsonify({
            "products": [],
            "response": "No image uploaded.",
            "extracted_text": ""
        })

    # Run EasyOCR
    extracted_text = ocr.extract_text(image_file)

    if not extracted_text:
        return jsonify({
            "products": [],
            "response": "Could not extract readable text.",
            "extracted_text": ""
        })

    # Vector DB search
    vec = embedder.embed_text([extracted_text])[0]
    results = db.query(vec, top_k=5)

    products = [{
        "id": m.id,
        "score": float(m.score),
        "product_name": m.metadata.get("product_name", ""),
        "price": m.metadata.get("price", "")
    } for m in results.matches]

    return jsonify({
        "products": products,
        "response": f"OCR text '{extracted_text}' matched these products.",
        "extracted_text": extracted_text
    })


# --------------------------------------------------
# MODULE 3 — CNN Image Product Search
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

    # CNN predict
    prediction = cnn.predict(temp_path)
    predicted_class = prediction["class"]

    # Search vectors
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
# START APP
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
