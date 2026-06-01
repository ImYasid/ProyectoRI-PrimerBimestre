import math
from src.models.base_model import BaseRetrievalModel

class TfidfIR(BaseRetrievalModel):
    def __init__(self, preprocessor, indexer):
        self.preprocessor = preprocessor
        self.indexer = indexer
        self.idf = {}
        self.doc_norms = {}

    def index(self):
        N = self.indexer.total_docs

        for term, docs in self.indexer.inverted_index.items():
            self.idf[term] = math.log(N / len(docs))

        for term, docs in self.indexer.inverted_index.items():
            for doc_id, tf in docs.items():
                peso = (1 + math.log(tf)) * self.idf[term]
                self.doc_norms[doc_id] = self.doc_norms.get(doc_id, 0.0) + peso ** 2

        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])

    def search(self, query: str, k: int = 5) -> list:
        tokens = self.preprocessor.transform(query).split()
        if not tokens:
            return []

        query_tf = {}
        for t in tokens:
            if t in self.idf:
                query_tf[t] = query_tf.get(t, 0) + 1

        if not query_tf:
            return []

        query_vec = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in query_tf.items()}
        query_norm = math.sqrt(sum(p ** 2 for p in query_vec.values()))

        scores = {}
        for term, q_peso in query_vec.items():
            for doc_id, tf in self.indexer.get_documents(term).items():
                doc_peso = (1 + math.log(tf)) * self.idf[term]
                scores[doc_id] = scores.get(doc_id, 0.0) + q_peso * doc_peso

        resultados = [(doc_id, s / (query_norm * self.doc_norms.get(doc_id, 1.0)))
                      for doc_id, s in scores.items()]
        return sorted(resultados, key=lambda x: x[1], reverse=True)[:k]
