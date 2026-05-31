from abc import ABC, abstractmethod

class BaseRetrievalModel(ABC):
    @abstractmethod
    def index(self):
        """
        Método para procesar y preparar las estructuras matemáticas 
        específicas del modelo a partir del corpus o el indexer.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list:
        """
        Método para buscar y retornar el top K de documentos más relevantes.
        Debe retornar una lista de tuplas: [(doc_id, score), ...]
        """
        pass