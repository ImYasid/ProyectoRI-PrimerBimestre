import math
from src.models.base_model import BaseRetrievalModel
from src.preprocessor import TextPreprocessor
from src.indexer import Indexer

class TfidfIR(BaseRetrievalModel):
    def __init__(self, preprocessor: TextPreprocessor, indexer: Indexer):
        self.preprocessor = preprocessor
        self.indexer = indexer
        
        # Diccionario para almacenar el IDF de cada término en el vocabulario
        self.idf = {}
        # Diccionario para almacenar la norma (longitud vectorial) de cada documento
        self.doc_norms = {}

    def calculate_idf(self):
        """
        Calcula el IDF para cada término en el índice invertido.
        IDF(t) = log(N / df(t)) donde:
        - N es el número total de documentos
        - df(t) es el número de documentos que contienen el término t
        """
        N = self.indexer.total_docs
        if N == 0:
            print("El índice está vacío. Construye el indexer primero.")
            return

        for term, documents in self.indexer.inverted_index.items():
            df = len(documents) # En cuántos documentos aparece este término
            # Fórmula IDF suavizada estándar
            self.idf[term] = math.log(N / df)

        print(self.idf) # Para depuración: muestra los IDF calculados para cada término

    def calculate_doc_norms(self):
        """
        Calcula la norma (longitud vectorial) de cada documento para acelerar el cálculo de similitud.
        La norma se calcula como la raíz cuadrada de la suma de los cuadrados de los pesos TF-IDF de los términos en el documento.
        """
        for term, documents in self.indexer.inverted_index.items():
            idf_term = self.idf.get(term, 0.0)
            for doc_id, tf in documents.items():
                # TF logarítmico: 1 + log(tf)
                tf_log = 1 + math.log(tf)
                peso_tfidf = tf_log * idf_term
                
                # Sumamos el cuadrado del peso a la norma del documento
                self.doc_norms[doc_id] = self.doc_norms.get(doc_id, 0.0) + (peso_tfidf ** 2)

        # Terminamos de calcular la norma aplicando la raíz cuadrada a la suma
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])

    def index(self):
        """Para indexar el modelo TF-IDF, necesitamos calcular el IDF de cada término y la norma de cada documento."""
        self.calculate_idf()
        self.calculate_doc_norms()

        print(f"✓ Modelo TF-IDF indexado (IDF y normas calculadas).")
        print(f"Términos con IDF calculado: {len(self.idf)}")
        print(f"Documentos con norma calculada: {len(self.doc_norms)}")

    def search(self, query: str, k: int = 5) -> list:
        """
        Busca usando Similitud de Coseno entre el vector TF-IDF de la query
        y los vectores TF-IDF de los documentos.
        """
        # 1. Preprocesar la query y contar frecuencias (TF de la query)
        clean_query = self.preprocessor.transform(query)
        query_tokens = clean_query.split()
        
        if not query_tokens:
            return []
            
        query_tf = {}
        for token in query_tokens:
            # Solo consideramos tokens que existan en nuestro vocabulario (que tengan IDF)
            if token in self.idf:
                query_tf[token] = query_tf.get(token, 0) + 1
                
        if not query_tf:
            return [] # La query tenía palabras desconocidas
            
        # 2. Vectorizar la query (TF-IDF) y calcular su norma
        query_vector = {}
        query_norm_sq = 0.0
        
        for term, tf in query_tf.items():
            tf_log = 1 + math.log(tf)
            peso = tf_log * self.idf[term]
            query_vector[term] = peso
            query_norm_sq += peso ** 2
            
        query_norm = math.sqrt(query_norm_sq)

        # 3. Calcular el producto punto (A · B) solo con los documentos candidatos
        scores_punto = {}
        for term, query_peso in query_vector.items():
            documents = self.indexer.get_documents(term)
            for doc_id, doc_tf in documents.items():
                doc_tf_log = 1 + math.log(doc_tf)
                doc_peso = doc_tf_log * self.idf[term]
                
                # Acumulamos el producto punto
                scores_punto[doc_id] = scores_punto.get(doc_id, 0.0) + (query_peso * doc_peso)

        # 4. Calcular el coseno final dividiendo por las normas
        resultados = []
        for doc_id, producto_punto in scores_punto.items():
            doc_norm = self.doc_norms.get(doc_id, 1.0) # 1.0 para evitar división por 0 accidental
            coseno = producto_punto / (query_norm * doc_norm)
            resultados.append((doc_id, coseno))

        # 5. Ordenar de mayor a menor y retornar top K
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:k]