from flask import Flask, request, jsonify
from services.embeddings import EmbeddingService
from services.vector_db import VectorDB
from services.ocr_service import OCRService  # <-- NEW IMPORT for OCR

app = Flask(__name__)

# --------------------------------------------------
# CREATE THESE OBJECTS ONLY ONCE
# --------------------------------------------------
embedder = EmbeddingService()
db = VectorDB()
ocr = OCRService()   # <-- NEW OCR OBJECT


@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.form.get('query', '')

    if not query:
        return jsonify({
            "products": [],
            "response": "Please provide a product query."
        })

    # Convert query to a vector
    query_vector = embedder.embed_text([query])[0]

    # Search Pinecone
    results = db.query(query_vector, top_k=5)

    # Format products
    products = []
    for match in results.matches:
        products.append({
            "id": match.id,
            "score": float(match.score),
            "product_name": match.metadata.get("product_name", ""),
            "price": match.metadata.get("price", "")
        })

    natural_response = f"Here are the top {len(products)} recommendations for '{query}'."

    return jsonify({
        "products": products,
        "response": natural_response
    })


# --------------------------------------------------
# MODULE 2 — OCR HANDWRITTEN QUERY ENDPOINT
# --------------------------------------------------
@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    image_file = request.files.get('image_data')

    if image_file is None:
        return jsonify({
            "products": [],
            "response": "No image uploaded.",
            "extracted_text": ""
        })

    # 1. Extract text from the uploaded image
    extracted_text = ocr.extract_text(image_file)

    if not extracted_text:
        return jsonify({
            "products": [],
            "response": "Could not extract readable text.",
            "extracted_text": ""
        })

    # 2. Convert extracted text to vector
    query_vector = embedder.embed_text([extracted_text])[0]

    # 3. Search Pinecone
    results = db.query(query_vector, top_k=5)

    # Format results
    products = []
    for match in results.matches:
        products.append({
            "id": match.id,
            "score": float(match.score),
            "product_name": match.metadata.get("product_name", ""),
            "price": match.metadata.get("price", "")
        })

    natural_response = f"OCR text '{extracted_text}' matched these products."

    return jsonify({
        "products": products,
        "response": natural_response,
        "extracted_text": extracted_text
    })


if __name__ == '__main__':
    app.run(debug=True)
