from src.models.base_model import BaseRetrievalModel
from src.preprocessor import TextPreprocessor
from src.indexer import Indexer

class JaccardIR(BaseRetrievalModel):
    def __init__(self, preprocessor: TextPreprocessor, indexer: Indexer):
        self.preprocessor = preprocessor
        self.indexer = indexer
        # Guardaremos cuántas palabras ÚNICAS tiene cada documento (|B|)
        self.doc_unique_terms_count = {}

    def index(self):
        """
        Para Jaccard con vectores binarios, necesitamos saber cuántos 
        términos ÚNICOS tiene cada documento.
        Lo calculamos eficientemente leyendo el índice invertido.
        """
        self.doc_unique_terms_count = {}
        
        # Recorremos el índice invertido. 
        # Si un documento aparece en los postings de un término, sumamos 1 a su conteo de términos únicos.
        for term, postings in self.indexer.inverted_index.items():
            for doc_id in postings.keys():
                # No nos importa la frecuencia (postings[doc_id]), solo su existencia (vector binario)
                self.doc_unique_terms_count[doc_id] = self.doc_unique_terms_count.get(doc_id, 0) + 1
                
        print(f"✓ Modelo Jaccard indexado.")
        print(f"Documentos vectorizados (binarios): {len(self.doc_unique_terms_count)}")

    def search(self, query: str, k: int = 5) -> list:
        """
        Busca usando Similitud de Jaccard: |A ∩ B| / (|A| + |B| - |A ∩ B|)
        """
        # 1. Preprocesar la query
        clean_query = self.preprocessor.transform(query)
        # Convertimos a SET porque en Jaccard Binario no importa si el usuario repite palabras en la query
        query_terms = set(clean_query.split()) 
        
        if not query_terms:
            return []

        # 2. Encontrar la intersección (|A ∩ B|) para los documentos candidatos
        intersection_counts = {}
        for term in query_terms:
            postings = self.indexer.get_postings(term)
            # Para cada documento que contenga este término de la query, sumamos 1 a la intersección
            for doc_id in postings.keys():
                intersection_counts[doc_id] = intersection_counts.get(doc_id, 0) + 1

        # 3. Calcular el score de Jaccard SOLO para los documentos que tengan al menos 1 coincidencia
        scores = []
        size_A = len(query_terms) # |A|
        
        for doc_id, size_intersection in intersection_counts.items():
            size_B = self.doc_unique_terms_count.get(doc_id, 0) # |B|
            size_union = size_A + size_B - size_intersection    # |A ∪ B|
            
            jaccard_score = size_intersection / size_union
            scores.append((doc_id, jaccard_score))

        # 4. Ordenar de mayor a menor score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Retornar los top K documentos
        return scores[:k]