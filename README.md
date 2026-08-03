# ChefBot AI

Questo è il progetto finale del modulo 5 del master, dedicato a Computer Vision e NLP.

L'idea è quella di partire dalla foto di un piatto, provare a riconoscerlo e mostrare alcune informazioni che Food101 non contiene, come ingredienti, descrizione e valori nutrizionali. Ho aggiunto anche una ricerca testuale per richieste del tipo *vorrei qualcosa di fresco* oppure *cerco un dolce cremoso*.

Per non rendere l'addestramento troppo pesante ho lavorato su 12 classi di Food101: bruschetta, caprese salad, cheesecake, greek salad, lasagna, paella, panna cotta, pizza, ravioli, spaghetti bolognese, spaghetti carbonara e tiramisù.

## Parte di classificazione

Il modello usato è MobileNetV2 con i pesi ImageNet. Nel notebook `01_training_mobilenet_food101.ipynb` la rete viene usata prima come base congelata e successivamente vengono sbloccati gli ultimi livelli per il fine-tuning.

Il notebook comprende anche i grafici di accuracy e loss e la matrice di confusione, che mi serve soprattutto per controllare quali piatti vengono confusi tra loro.

## Knowledge base e ricerca

Le informazioni aggiuntive sui piatti sono nel file `data/knowledge_base.json`. Non provengono da Food101 e sono valori indicativi, soprattutto per la parte nutrizionale.

Nel secondo notebook ho provato la ricerca semantica. Le richieste vengono trasformate in embeddings con un modello Sentence Transformers multilingua e confrontate con le descrizioni dei piatti. In questo modo la ricerca non dipende dalla presenza della stessa parola nel testo.

## File principali

- `notebooks/01_training_mobilenet_food101.ipynb`: training e valutazione del classificatore
- `notebooks/02_ricerca_semantica.ipynb`: alcune prove della ricerca testuale
- `app.py`: interfaccia Streamlit
- `src/`: funzioni usate dall'app
- `data/knowledge_base.json`: schede dei 12 piatti

## Avvio

Per creare l'ambiente da PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Il dataset viene scaricato da TensorFlow Datasets quando si esegue il primo notebook.

Terminato l'addestramento, il modello viene salvato come `models/chefbot_mobilenet.keras`. A quel punto l'app può essere avviata dalla cartella principale:

```powershell
streamlit run app.py
```

Al momento il classificatore può scegliere solo tra le 12 classi usate nel training. Una foto appartenente a una categoria diversa verrebbe quindi associata comunque a uno dei piatti conosciuti.
