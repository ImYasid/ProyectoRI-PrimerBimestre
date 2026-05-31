import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)
        self.stemmer = SnowballStemmer(language)
        self.cache = {}
        
        
    def transform(self, texto):
        """
        Limpia, tokeniza y aplica stemming a un texto.
        """
        if not isinstance(texto, str):
            return []
        
        texto = texto.lower()  # ¡Importante! El stemmer y los modelos necesitan todo en minúsculas
        texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        # 2. Tokenización (Mantenemos la LISTA completa para conservar las frecuencias)
        palabras_tokenizadas = texto.split()  # Usamos split simple para mantener la estructura de tokens
        
        # 3. Stemming con optimización por caché
        tokens_con_stem = []
        for word in palabras_tokenizadas:
            # Si no está en el caché del objeto, lo calculamos y guardamos
            if word not in self.cache:
                self.cache[word] = self.stemmer.stem(word)
            # Agregamos el stem final
            tokens_con_stem.append(self.cache[word])    

        return ' '.join(tokens_con_stem)
