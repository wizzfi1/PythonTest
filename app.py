from flask import Flask, request, jsonify
from flask_cors import CORS

from services.embeddings import EmbeddingService
from services.vector_db import VectorDB
from services.ocr_service import EasyOCRService

import os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# INITIALIZE ONLY LIGHTWEIGHT SERVICES
# --------------------------------------------------
embedder = EmbeddingService()
db = VectorDB()
ocr = EasyOCRService()     # OCR that works in Codespaces

cnn = None                 # CNN is lazy-loaded later


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

    query_vector = embedder.embed_text([query])[0]
    results = db.query(query_vector, top_k=5)

    products = [
        {
            "id": m.id,
            "score": float(m.score),
            "product_name": m.metadata.get("product_name", ""),
            "price": m.metadata.get("price", "")
        }
        for m in results.matches
    ]

    return jsonify({
        "products": products,
        "response": f"Here are the top {len(products)} recommendations for '{query}'."
    })


# --------------------------------------------------
# MODULE 2 — EASYOCR HANDWRITTEN OCR
# --------------------------------------------------
@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    image_file = request.files.get("image_data")

    if image_file is None:
        return jsonify({
            "products": [],
            "extracted_text": "",
            "response": "No image uploaded."
        })

    extracted_text = ocr.extract_text(image_file)

    if not extracted_text or extracted_text.strip() == "":
        return jsonify({
            "products": [],
            "extracted_text": "",
            "response": "Could not extract readable text."
        })

    # Query Pinecone with extracted OCR text
    query_vector = embedder.embed_text([extracted_text])[0]
    results = db.query(query_vector, top_k=5)

    products = [
        {
            "id": m.id,
            "score": float(m.score),
            "product_name": m.metadata.get("product_name", ""),
            "price": m.metadata.get("price", "")
        }
        for m in results.matches
    ]

    return jsonify({
        "products": products,
        "extracted_text": extracted_text,
        "response": f"OCR text '{extracted_text}' matched these products."
    })


# --------------------------------------------------
# MODULE 3 — LAZY LOADED CNN IMAGE CLASSIFICATION
# --------------------------------------------------
@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    global cnn

    # Lazy-load CNN ONLY when endpoint is first used
    if cnn is None:
        from services.cnn_predict import CNNPredictor
        cnn = CNNPredictor(
            model_path="models/cnn_transfer.keras",
            class_json="models/class_names.json"
        )

    file = request.files.get("product_image")
    if file is None:
        return jsonify({
            "class": "",
            "products": [],
            "response": "No image uploaded."
        })

    # Save temporary image
    temp_path = "tmp_upload.jpg"
    file.save(temp_path)

    # CNN prediction
    prediction = cnn.predict(temp_path)
    predicted_class = prediction["class"]

    # Search Pinecone for similar products
    query_vector = embedder.embed_text([predicted_class])[0]
    results = db.query(query_vector, top_k=5)

    products = [
        {
            "id": m.id,
            "score": float(m.score),
            "product_name": m.metadata.get("product_name", ""),
            "price": m.metadata.get("price", "")
        }
        for m in results.matches
    ]

    return jsonify({
        "class": predicted_class,
        "products": products,
        "response": f"Predicted class '{predicted_class}'. Here are similar products."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
