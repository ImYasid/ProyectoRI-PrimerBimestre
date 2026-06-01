import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    def transform(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        tokens = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words]
        return ' '.join(tokens)
