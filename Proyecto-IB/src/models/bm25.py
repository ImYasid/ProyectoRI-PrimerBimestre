import math
from src.models.base_model import BaseRetrievalModel

class BM25IR(BaseRetrievalModel):
    def __init__(self, preprocessor, indexer, k1=1.5, b=0.75):
        self.preprocessor = preprocessor
        self.indexer = indexer
        self.k1 = k1
        self.b = b
        self.idf = {}

    def index(self):
        N = self.indexer.total_docs
        for term, docs in self.indexer.inverted_index.items():
            df = len(docs)
            self.idf[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list:
        tokens = self.preprocessor.transform(query).split()
        if not tokens:
            return []

        scores = {}
        avgdl = self.indexer.avg_doc_length

        for term in tokens:
            if term not in self.idf:
                continue
            for doc_id, tf in self.indexer.get_documents(term).items():
                dl = self.indexer.doc_lengths.get(doc_id, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + self.idf[term] * (num / den)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
