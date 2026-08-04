import json
from pathlib import Path

import keras
import numpy as np
from PIL import Image


IMAGE_SIZE = (224, 224)


class FoodClassifier:
    """Carica il modello addestrato e classifica una fotografia."""

    def __init__(
        self,
        model_path="models/chefbot_mobilenet.keras",
        classes_path="models/class_names.json",
    ):
        model_file = Path(model_path)
        classes_file = Path(classes_path)

        if not model_file.exists():
            raise FileNotFoundError(
                "Modello non trovato. Eseguire prima il notebook di training."
            )
        if not classes_file.exists():
            raise FileNotFoundError("File delle classi non trovato.")

        self.model = keras.models.load_model(model_file)

        with classes_file.open("r", encoding="utf-8") as file:
            self.class_names = json.load(file)

    def preprocess(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def predict(self, image, top_k=3):
        image_array = self.preprocess(image)
        probabilities = self.model.predict(image_array, verbose=0)[0]
        best_indexes = np.argsort(probabilities)[::-1][:top_k]

        return [
            {
                "label": self.class_names[index],
                "confidence": float(probabilities[index]),
            }
            for index in best_indexes
        ]
