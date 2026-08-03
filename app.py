from pathlib import Path

import streamlit as st
from PIL import Image

from src.classifier import FoodClassifier
from src.knowledge_base import get_dish_by_label, load_knowledge_base
from src.semantic_search import DishSearch


st.set_page_config(page_title="ChefBot AI", page_icon="🍽️", layout="wide")

DISHES = load_knowledge_base()


@st.cache_resource
def load_classifier():
    return FoodClassifier()


@st.cache_resource
def load_search_engine():
    return DishSearch(DISHES)


def show_dish_card(dish):
    st.subheader(dish["name"])
    st.write(dish["description"])

    left, right = st.columns(2)

    with left:
        st.markdown("**Ingredienti principali**")
        for ingredient in dish["ingredients"]:
            st.write(f"- {ingredient}")

    with right:
        nutrition = dish["nutrition"]
        st.markdown(f'**Valori indicativi per {nutrition["portion"]}**')
        st.write(f'Calorie: {nutrition["calories_kcal"]} kcal')
        st.write(f'Proteine: {nutrition["protein_g"]} g')
        st.write(f'Carboidrati: {nutrition["carbohydrates_g"]} g')
        st.write(f'Grassi: {nutrition["fat_g"]} g')


st.title("ChefBot AI")
st.write(
    "Carica la foto di un piatto oppure descrivi ciò che vorresti mangiare."
)

recognition_tab, search_tab = st.tabs(
    ["Riconoscimento da immagine", "Ricerca per descrizione"]
)

with recognition_tab:
    uploaded_file = st.file_uploader(
        "Scegli una fotografia",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_column, result_column = st.columns([1, 1])

        with image_column:
            st.image(image, caption="Immagine caricata", use_container_width=True)

        try:
            classifier = load_classifier()
            predictions = classifier.predict(image, top_k=3)
            best_prediction = predictions[0]
            dish = get_dish_by_label(best_prediction["label"], DISHES)

            with result_column:
                st.success(
                    f'Piatto riconosciuto: {dish["name"]} '
                    f'({best_prediction["confidence"]:.1%})'
                )
                show_dish_card(dish)

                with st.expander("Altre ipotesi del modello"):
                    for prediction in predictions[1:]:
                        alternative = get_dish_by_label(
                            prediction["label"],
                            DISHES,
                        )
                        st.write(
                            f'{alternative["name"]}: '
                            f'{prediction["confidence"]:.1%}'
                        )

        except FileNotFoundError as error:
            with result_column:
                st.warning(str(error))
                st.info(
                    "Il riconoscimento sarà disponibile dopo aver eseguito "
                    "il notebook di addestramento."
                )

with search_tab:
    query = st.text_input(
        "Che tipo di piatto cerchi?",
        placeholder="Ad esempio: vorrei qualcosa di fresco e leggero",
    )

    if st.button("Cerca il piatto", type="primary"):
        if not query.strip():
            st.warning("Scrivi prima una breve descrizione.")
        else:
            with st.spinner("Confronto la richiesta con i piatti disponibili..."):
                search_engine = load_search_engine()
                results = search_engine.search(query, top_k=3)

            best_result = results[0]
            show_dish_card(best_result["dish"])

            st.caption(
                f'Similarità semantica: {best_result["score"]:.3f}'
            )

            with st.expander("Altri piatti simili"):
                for result in results[1:]:
                    st.write(
                        f'{result["dish"]["name"]} '
                        f'- similarità {result["score"]:.3f}'
                    )

st.divider()
st.caption(
    "I valori nutrizionali sono stime indicative e possono cambiare "
    "in base alla ricetta e alla porzione."
)
