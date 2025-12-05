import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "product-index")

pc = Pinecone(api_key=api_key)

# List existing indexes
existing_indexes = pc.list_indexes().names()
print("Existing indexes:", existing_indexes)

# POD-BASED INDEX CREATION for region us-east-1
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,       # MiniLM model output size
        metric="cosine",
        pods=1,
        replicas=1,
        pod_type="p1.x1",    # cheapest pod type
        region="us-east-1"
    )
    print(f"Created Pinecone index (pod-based): {index_name}")
else:
    print(f"Index already exists: {index_name}")
