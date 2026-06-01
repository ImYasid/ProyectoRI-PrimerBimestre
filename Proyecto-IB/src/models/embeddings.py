import chromadb
from sentence_transformers import SentenceTransformer
from src.models.base_model import BaseRetrievalModel

class SemanticIR(BaseRetrievalModel):
    def __init__(self, df_corpus, text_column='text', id_column='doc_id', model_name='all-MiniLM-L6-v2'):
        self.df = df_corpus
        self.text_col = text_column
        self.id_col = id_column
        self.encoder = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client()

        try:
            self.chroma_client.delete_collection("corpus_embeddings")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="corpus_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def index(self, batch_size=500):
        docs = self.df[self.text_col].tolist()
        ids = self.df[self.id_col].tolist()

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            embeddings = self.encoder.encode(batch_docs).tolist()
            self.collection.add(embeddings=embeddings, documents=batch_docs, ids=batch_ids)

    def search(self, query: str, k: int = 5) -> list:
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)

        formatted = []
        if results['ids']:
            for doc_id, dist in zip(results['ids'][0], results['distances'][0]):
                formatted.append((doc_id, 1.0 - dist))
        return formatted
