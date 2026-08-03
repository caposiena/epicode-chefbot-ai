import numpy as np
from sentence_transformers import SentenceTransformer

from src.knowledge_base import dish_to_text


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DishSearch:
    """Ricerca dei piatti tramite similarità tra embeddings."""

    def __init__(self, dishes, model_name=MODEL_NAME):
        self.dishes = dishes
        self.model = SentenceTransformer(model_name)
        self.texts = [dish_to_text(dish) for dish in dishes]
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
        )

    def search(self, query, top_k=3):
        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # Con vettori normalizzati il prodotto scalare coincide
        # con la cosine similarity.
        scores = self.embeddings @ query_embedding
        best_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "dish": self.dishes[index],
                "score": float(scores[index]),
            }
            for index in best_indexes
        ]
