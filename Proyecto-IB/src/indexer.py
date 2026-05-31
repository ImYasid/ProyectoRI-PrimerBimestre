from src.preprocessor import TextPreprocessor

class Indexer:
    def __init__(self):
        # El índice invertido: { término: { doc_id: frecuencia } }
        self.inverted_index = {}
        # Diccionario para almacenar la longitud de cada documento { doc_id: longitud }
        self.doc_lengths = {}
        # Variables usadas para el modelo BM25
        self.avg_doc_length = 0.0
        self.total_docs = 0

    def build_index(self, df, clean_text_column: str, id_column: str):
        """
        Construye el índice invertido a partir de la columna ya preprocesada y limpia.
        """
        # Asegurar que limpiamos datos previos si se vuelve a ejecutar
        self.inverted_index = {}
        self.doc_lengths = {}
        
        # Filtrar nulos para evitar errores
        self.total_docs = len(df)
        total_words_corpus = 0

        # Iteramos eficientemente sobre el DataFrame
        for _, row in df.iterrows():
            doc_id = row[id_column]
            clean_content = row[clean_text_column]
            
            # Como la función transform ya devuelve un string limpio unido por espacios,
            # un split() simple nos recupera la lista exacta de tokens con stem.
            tokens = clean_content.split()
            
            # Registrar la longitud de este documento
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_words_corpus += doc_len
            
            # Construcción del índice y conteo de frecuencias locales (TF)
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                
                # Sumamos 1 a la frecuencia del término en este documento específico
                current_freq = self.inverted_index[token].get(doc_id, 0)
                self.inverted_index[token][doc_id] = current_freq + 1

        # Calcular la longitud promedio del corpus (parámetro obligatorio para BM25)
        if self.total_docs > 0:
            self.avg_doc_length = total_words_corpus / self.total_docs
            
        print("✓ Índice invertido construido con éxito a partir de texto preprocesado.")
        print(f"Total de documentos indexados: {self.total_docs}")
        print(f"Número de términos únicos en el índice: {len(self.inverted_index)}")
    
    def get_documents(self, term: str) -> dict:
        """
        Retorna los documentos y sus respectivas frecuencias para un término.
        Si el término no existe, devuelve un diccionario vacío.
        """
        return self.inverted_index.get(term, {})