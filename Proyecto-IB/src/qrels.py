import pandas as pd

class QrelsManager:
    """
    Clase para gestionar, generar y formatear qrels (relevance judgments) 
    a partir de múltiples modelos de recuperación de información o desde un corpus etiquetado.
    """
    
    def __init__(self, modelos_dict, queries_dict):
        """
        Inicializa el gestor de Qrels.
        
        :param modelos_dict: Diccionario de modelos de recuperación instanciados.
        :param queries_dict: Diccionario de consultas con formato {q_id: q_text}.
        """
        self.modelos_dict = modelos_dict
        self.queries_dict = queries_dict
        self.qrels_detallado = {}

    def generar_qrels_por_modelos(self, k_pool=3):
        """
        Genera un diccionario de qrels donde cada query contiene un sub-diccionario
        con los documentos recuperados por cada modelo y lo guarda en el estado de la clase.
        Útil para inspección visual y debugging de los modelos.
        """
        self.qrels_detallado = {}
        
        print(f"Generando Qrels Detallados con k={k_pool}...")
        
        for q_id, q_text in self.queries_dict.items():
            self.qrels_detallado[q_id] = {}
            
            # Consultamos cada modelo de forma individual
            for nombre, modelo in self.modelos_dict.items():
                resultados = modelo.search(q_text, k=k_pool)
                
                # Extraemos solo los IDs de los documentos
                docs_recuperados = [doc_id for doc_id, _ in resultados]
                
                # Almacenamos la lista bajo el nombre del modelo
                self.qrels_detallado[q_id][nombre] = docs_recuperados
                
            print(f"[{q_id}] '{q_text}' procesada en todos los modelos.")
            
        return self.qrels_detallado

    def a_dataframe_comparativo(self):
        """
        Transforma el diccionario estructurado de qrels actual en un DataFrame plano
        donde se alinean los documentos recuperados por cada modelo por posición (Rank).
        """
        if not self.qrels_detallado:
            raise ValueError("Los qrels detallados están vacíos. Ejecuta 'generar_qrels_por_modelos' primero.")
            
        filas = []
        
        for q_id, modelos_docs in self.qrels_detallado.items():
            q_text = self.queries_dict.get(q_id, "Consulta no encontrada")
            
            # Encontrar el número máximo de resultados devueltos entre todos los modelos
            max_k = max(len(docs) for docs in modelos_docs.values()) if modelos_docs else 0
            
            for i in range(max_k):
                fila = {
                    "Query ID": q_id,
                    "Consulta": q_text,
                    "Rank": i + 1
                }
                
                # Asignar el documento correspondiente a cada modelo para el Rank actual
                for nombre_modelo, lista_docs in modelos_docs.items():
                    if i < len(lista_docs):
                        fila[nombre_modelo] = lista_docs[i]
                    else:
                        fila[nombre_modelo] = "N/A" # En caso de que un modelo devuelva menos resultados
                
                filas.append(fila)
                
        return pd.DataFrame(filas)

    def generar_ground_truth_desde_corpus(self, df_corpus, mapeo_categorias):
        """
        NUEVA FUNCIÓN: Genera el diccionario de qrels reales (Ground Truth) buscando en el 
        DataFrame qué documentos pertenecen a la categoría correcta de cada consulta.
        ESTE ES EL DICCIONARIO QUE DEBE PASARSE AL EVALUATOR.
        
        :param df_corpus: DataFrame que contiene las columnas 'doc_id' y 'categoria'.
        :param mapeo_categorias: Diccionario que relaciona el Query ID con la etiqueta del dataset.
        :return: Diccionario plano de qrels {q_id: [doc_id1, doc_id2, ...]}
        """
        qrels_reales = {}
        
        print("\n--- Extrayendo Terreno de Verdad (Ground Truth) del Corpus ---")
        for q_id, categoria_buscada in mapeo_categorias.items():
            # Filtramos el corpus buscando la categoría. 
            filtro = df_corpus['categoria'].astype(str).str.contains(categoria_buscada, case=False, na=False)
            
            # Extraemos los IDs de los documentos que cumplen la condición
            docs_relevantes = df_corpus[filtro]['doc_id'].tolist()
            
            qrels_reales[q_id] = docs_relevantes
            
            print(f"[{q_id}] Categoría '{categoria_buscada}': {len(docs_relevantes)} documentos reales encontrados.")
            
        return qrels_reales