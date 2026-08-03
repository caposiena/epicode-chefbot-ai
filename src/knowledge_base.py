import json
from pathlib import Path


def load_knowledge_base(path="data/knowledge_base.json"):
    """Carica le schede dei piatti dal file JSON."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base non trovata: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        dishes = json.load(file)

    return dishes


def get_dish_by_label(label, dishes):
    """Restituisce la scheda associata alla classe Food101."""
    for dish in dishes:
        if dish["label"] == label:
            return dish
    return None


def dish_to_text(dish):
    """Crea il testo utilizzato per il confronto semantico."""
    ingredients = ", ".join(dish["ingredients"])

    return (
        f'{dish["name"]}. {dish["description"]} '
        f'Ingredienti principali: {ingredients}. '
        f'Caratteristiche: {dish["semantic_tags"]}.'
    )
