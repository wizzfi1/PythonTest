from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text_list):
        """
        Input: list of strings
        Output: list of vectors (384 dimensions each)
        """
        embeddings = self.model.encode(text_list, convert_to_numpy=True)
        return embeddings.tolist()
