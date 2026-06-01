import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.preprocessor import TextPreprocessor
from src.indexer import Indexer
from src.models.tfidf import TfidfIR
from src.models.jaccard import JaccardIR
from src.models.bm25 import BM25IR

def cargar_sistema():
    df = pd.read_csv(os.path.join("data", "ModApte_train.csv"))
    df['doc_id'] = [f"Document_{i}" for i in range(len(df))]
    df['text'] = df['text'].fillna("")

    preprocessor = TextPreprocessor(language='english')
    df['text_clean'] = df['text'].apply(preprocessor.transform)

    indexer = Indexer()
    indexer.build_index(df, clean_text_column='text_clean', id_column='doc_id')

    jaccard = JaccardIR(preprocessor, indexer); jaccard.index()
    tfidf   = TfidfIR(preprocessor, indexer);   tfidf.index()
    bm25    = BM25IR(preprocessor, indexer);    bm25.index()

    return df, jaccard, tfidf, bm25

def mostrar_resultados(resultados, df, k):
    if not resultados:
        print("Sin resultados.")
        return
    print(f"\nTop {k} resultados:")
    for rank, (doc_id, score) in enumerate(resultados, 1):
        snippet = df.loc[df['doc_id'] == doc_id, 'text'].values[0][:100]
        print(f"[{rank}] {doc_id} | Score: {score:.4f} | {snippet}...")

def main():
    df, jaccard, tfidf, bm25 = cargar_sistema()
    modelos = {'1': ('Jaccard', jaccard), '2': ('TF-IDF', tfidf), '3': ('BM25', bm25)}

    while True:
        print("\n1. Jaccard  2. TF-IDF  3. BM25  4. Salir")
        opcion = input("Modelo: ")
        if opcion == '4':
            break
        if opcion not in modelos:
            continue
        query = input("Consulta: ")
        nombre, modelo = modelos[opcion]
        print(f"\nModelo: {nombre}")
        mostrar_resultados(modelo.search(query), df, k=5)

if __name__ == "__main__":
    main()
