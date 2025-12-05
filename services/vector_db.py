import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

class VectorDB:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX", "product-index")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upsert_products(self, product_vectors):
        """
        product_vectors = [
          {"id": "product_id", "values": [...], "metadata": {...}}
        ]
        """
        self.index.upsert(vectors=product_vectors)
        print(f"Upserted {len(product_vectors)} products.")

    def query(self, vector, top_k=5):
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        return result
