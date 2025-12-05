import pandas as pd
from services.embeddings import EmbeddingService
from services.vector_db import VectorDB

# Load your product catalog
df = pd.read_csv("data/product_catalog.csv")

# Create the text we will embed
texts = df["product_name"].astype(str).tolist()

embedder = EmbeddingService()
vectors = embedder.embed_text(texts)

# Prepare Pinecone upsert format
items = []
for product_id, vector, price, name in zip(df["product_id"], vectors, df["price"], df["product_name"]):
    metadata = {
        "product_name": name,
        "price": price
    }
    items.append({
        "id": str(product_id),
        "values": vector,
        "metadata": metadata
    })

# Upload in batches
db = VectorDB()

batch_size = 100  # safe size
total = len(items)

print(f"Uploading {total} vectors in batches of {batch_size}...")

for i in range(0, total, batch_size):
    batch = items[i:i + batch_size]
    db.upsert_products(batch)
    print(f"Uploaded batch {i} → {i+len(batch)}")

print("All vectors uploaded successfully!")
