# Proyecto de Recuperación de Información

Sistema de búsqueda de documentos usando el corpus Reuters ModApte. Se implementaron 4 modelos distintos para comparar cuál encuentra mejor los documentos relevantes.

---

## Estructura

```
Proyecto-IB/
├── main.py                  # acá se corre todo
├── informe_tecnico.ipynb    # notebook con experimentos
├── data/
│   └── ModApte_train.csv    # el corpus (hay que ponerlo a mano)
└── src/
    ├── preprocessor.py      # limpia el texto
    ├── indexer.py           # construye el índice invertido
    ├── evaluator.py         # calcula precisión, recall y MAP
    └── models/
        ├── base_model.py    # clase base abstracta
        ├── jaccard.py
        ├── tfidf.py
        ├── bm25.py
        └── embeddings.py   # modelo semántico (opcional)
```

---

## Modelos implementados

| Modelo | Idea básica |
|--------|-------------|
| **Jaccard** | compara cuántos términos comparten la consulta y el doc |
| **TF-IDF** | le da más peso a palabras raras e importantes |
| **BM25** | versión mejorada de TF-IDF, penaliza docs muy largos |
| **Semántico** | usa embeddings para entender el significado (requiere instalar más cosas) |

---

## Cómo instalar

```bash
pip install pandas nltk
python -c "import nltk; nltk.download('stopwords')"
```

Si también querés usar el modelo semántico:

```bash
pip install sentence-transformers chromadb
```

Después colocá el archivo `ModApte_train.csv` dentro de la carpeta `data/`.

---

## Cómo ejecutar

```bash
python main.py
```

Te va a aparecer un menú así:

```
1. Jaccard  2. TF-IDF  3. BM25  4. Salir
Modelo: 3
Consulta: oil prices

Top 5 resultados:
[1] Document_42 | Score: 14.38 | Saudi Arabia announced...
[2] Document_107 | Score: 12.95 | OPEC ministers gathered...
```

Elegís el modelo, escribís tu consulta y listo.

---

## Requisitos

- Python 3.9+
- pandas, nltk
- (opcional) sentence-transformers, chromadb
