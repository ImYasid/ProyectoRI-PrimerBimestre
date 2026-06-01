from src.models.base_model import BaseRetrievalModel

class JaccardIR(BaseRetrievalModel):
    def __init__(self, preprocessor, indexer):
        self.preprocessor = preprocessor
        self.indexer = indexer
        self.doc_unique_terms = {}

    def index(self):
        for term, docs in self.indexer.inverted_index.items():
            for doc_id in docs:
                self.doc_unique_terms[doc_id] = self.doc_unique_terms.get(doc_id, 0) + 1

    def search(self, query: str, k: int = 5) -> list:
        query_terms = set(self.preprocessor.transform(query).split())
        if not query_terms:
            return []

        intersecciones = {}
        for term in query_terms:
            for doc_id in self.indexer.get_documents(term):
                intersecciones[doc_id] = intersecciones.get(doc_id, 0) + 1

        resultados = []
        for doc_id, inter in intersecciones.items():
            union = len(query_terms) + self.doc_unique_terms.get(doc_id, 0) - inter
            score = inter / union if union > 0 else 0
            resultados.append((doc_id, score))

        return sorted(resultados, key=lambda x: x[1], reverse=True)[:k]
