import numpy as np
from sentence_transformers import SentenceTransformer

from src.knowledge_base import dish_to_text


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DishSearch:
    """Ricerca dei piatti tramite similarità tra embeddings."""

    def __init__(self, dishes, model_name=MODEL_NAME):
        self.dishes = dishes
        self.model = SentenceTransformer(model_name)

        self.texts = []
        self.text_dish_indexes = []

        for dish_index, dish in enumerate(dishes):
            candidate_texts = [
                dish_to_text(dish),
                *dish.get("search_examples", []),
            ]

            self.texts.extend(candidate_texts)
            self.text_dish_indexes.extend(
                [dish_index] * len(candidate_texts)
            )

        self.text_dish_indexes = np.array(self.text_dish_indexes)
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

        # Ogni piatto ha una descrizione e alcune richieste di esempio.
        # Come punteggio del piatto uso il confronto migliore.
        text_scores = self.embeddings @ query_embedding
        dish_scores = np.array(
            [
                text_scores[self.text_dish_indexes == dish_index].max()
                for dish_index in range(len(self.dishes))
            ]
        )

        best_indexes = np.argsort(dish_scores)[::-1][:top_k]

        return [
            {
                "dish": self.dishes[index],
                "score": float(dish_scores[index]),
            }
            for index in best_indexes
        ]
