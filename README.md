# ChefBot AI

Progetto finale del modulo 5 del Master in Python, AI e Machine Learning.

L'obiettivo è riconoscere un piatto a partire da una fotografia e collegare la previsione a una piccola base di conoscenza. Il progetto comprende anche una ricerca semantica: l'utente può descrivere ciò che vorrebbe mangiare, per esempio "qualcosa di fresco e leggero", e ricevere il piatto più simile tra quelli disponibili.

## Classi utilizzate

Per limitare i tempi di addestramento ho selezionato 12 categorie di Food101:

- bruschetta
- caprese salad
- cheesecake
- greek salad
- lasagna
- paella
- panna cotta
- pizza
- ravioli
- spaghetti bolognese
- spaghetti carbonara
- tiramisù

La selezione comprende primi piatti, piatti freschi e dessert. Alcune classi sono abbastanza simili tra loro e permettono di osservare gli errori del classificatore nella matrice di confusione.

## Struttura del progetto

```text
epicode-chefbot-ai/
├── data/
│   └── knowledge_base.json
├── models/
│   ├── class_names.json
│   └── README.md
├── notebooks/
│   ├── 01_training_mobilenet_food101.ipynb
│   └── 02_ricerca_semantica.ipynb
├── src/
│   ├── classifier.py
│   ├── knowledge_base.py
│   └── semantic_search.py
├── app.py
└── requirements.txt
```

Il primo notebook contiene il caricamento di Food101, il transfer learning, il fine-tuning e la valutazione. Il secondo serve a controllare separatamente il funzionamento della ricerca semantica.

## Come funziona

### Riconoscimento dell'immagine

Il classificatore utilizza MobileNetV2 con pesi ImageNet. Durante la prima fase la base convoluzionale rimane congelata e vengono addestrati soltanto i nuovi livelli finali. In seguito vengono sbloccati gli ultimi 20 livelli per un breve fine-tuning con learning rate più basso.

Il modello riceve immagini RGB ridimensionate a 224 x 224 pixel e restituisce una probabilità per ognuna delle 12 classi.

### Knowledge base

Food101 mette a disposizione immagini ed etichette, ma non contiene ingredienti o valori nutrizionali. Queste informazioni sono state inserite manualmente nel file `data/knowledge_base.json`.

Ogni scheda contiene:

- nome del piatto;
- breve descrizione;
- ingredienti principali;
- valori nutrizionali indicativi;
- alcune caratteristiche utili per la ricerca.

### Ricerca semantica

Le descrizioni dei piatti e la richiesta dell'utente vengono trasformate in embeddings con il modello multilingua `paraphrase-multilingual-MiniLM-L12-v2`.

Gli embeddings sono normalizzati e confrontati tramite prodotto scalare, che in questo caso corrisponde alla cosine similarity. Non è quindi necessario che la richiesta contenga le stesse parole presenti nella scheda.

## Installazione

Da PowerShell, dopo aver scaricato la repository:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Food101 viene scaricato automaticamente da TensorFlow Datasets quando si esegue il notebook di training. Il download e l'addestramento possono richiedere tempo.

## Addestramento

Aprire ed eseguire in ordine:

```text
notebooks/01_training_mobilenet_food101.ipynb
```

Alla fine del notebook viene creato il file:

```text
models/chefbot_mobilenet.keras
```

Non ho inserito un modello già addestrato perché le metriche riportate nel progetto devono derivare da un'esecuzione reale.

## Avvio dell'app

Dalla cartella principale del progetto:

```powershell
streamlit run app.py
```

L'app contiene due sezioni:

1. caricamento di una fotografia e visualizzazione della scheda del piatto riconosciuto;
2. ricerca di un piatto attraverso una descrizione libera.

Se il modello non è ancora presente, la ricerca semantica resta utilizzabile e la sezione di riconoscimento mostra un avviso.

## Limiti

Il classificatore conosce soltanto le 12 classi selezionate. Se viene caricata la foto di un piatto diverso, sceglierà comunque una delle categorie conosciute.

Anche calorie e nutrienti sono valori medi indicativi: possono cambiare in base alla ricetta, alla quantità degli ingredienti e alla porzione.

Un possibile miglioramento sarebbe aggiungere una soglia minima di confidenza, aumentare il numero di piatti e verificare la ricerca semantica su un insieme di richieste raccolte da utenti reali.
