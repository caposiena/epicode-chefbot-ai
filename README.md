CHEFBOT AI

Progetto finale del modulo 5 del Master in Python, AI e Machine Learning, dedicato alla Computer Vision e all'elaborazione del linguaggio naturale.

L'idea del progetto è realizzare una piccola applicazione in grado di riconoscere un piatto partendo da una fotografia. Dopo il riconoscimento, ChefBot mostra una scheda con una descrizione del piatto, gli ingredienti principali e una stima dei valori nutrizionali. Queste informazioni non sono presenti in Food101 e sono state quindi raccolte in una knowledge base realizzata in formato JSON.

Per mantenere l'addestramento gestibile ho selezionato dodici categorie del dataset: bruschetta, caprese salad, cheesecake, greek salad, lasagna, paella, panna cotta, pizza, ravioli, spaghetti bolognese, spaghetti carbonara e tiramisù. La scelta comprende piatti abbastanza diversi, ma anche alcune categorie simili tra loro. Questo permette di osservare nella matrice di confusione quali immagini creano più difficoltà al modello.

Per la classificazione ho utilizzato MobileNetV2 con i pesi già appresi su ImageNet. Nel primo notebook la parte convoluzionale viene inizialmente congelata e vengono addestrati i nuovi livelli finali. In una seconda fase vengono sbloccati gli ultimi venti livelli della rete per effettuare un breve fine-tuning con un learning rate più basso. Al termine vengono calcolate le metriche sul test set, viene costruita la matrice di confusione e il modello viene salvato nella cartella models.

La seconda parte del progetto riguarda la ricerca semantica. L'utente può scrivere una frase come “vorrei qualcosa di fresco e leggero” oppure “cerco un dolce cremoso”. La frase e le descrizioni dei piatti vengono trasformate in embeddings attraverso un modello Sentence Transformers multilingua. Il confronto viene eseguito usando la cosine similarity e restituisce i piatti semanticamente più vicini alla richiesta, senza limitarsi alla ricerca di parole identiche.

Il notebook 01_training_mobilenet_food101.ipynb contiene il caricamento di Food101, il transfer learning, il fine-tuning e la valutazione del classificatore. Il notebook 02_ricerca_semantica.ipynb contiene alcune richieste di prova. Il file app.py riunisce le due funzioni in una semplice interfaccia Streamlit, mentre la cartella src contiene le funzioni usate dall'applicazione.

Per installare il progetto è necessario creare un ambiente virtuale, attivarlo e installare le librerie indicate nel file requirements.txt. Da PowerShell i comandi sono: python -m venv .venv, .venv\Scripts\activate e pip install -r requirements.txt.

Food101 viene scaricato automaticamente da TensorFlow Datasets durante la prima esecuzione del notebook. Dopo l'addestramento viene creato il file models/chefbot_mobilenet.keras. A quel punto l'applicazione può essere avviata dalla cartella principale con il comando streamlit run app.py.

Il classificatore è specializzato soltanto sulle dodici categorie scelte. Se riceve la fotografia di un piatto diverso, prova comunque ad associarla a una delle classi che conosce. Anche i valori nutrizionali inseriti nelle schede sono indicativi, perché possono cambiare in base alla ricetta e alla quantità degli ingredienti.
