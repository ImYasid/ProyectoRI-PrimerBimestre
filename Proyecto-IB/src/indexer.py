class Indexer:
    def __init__(self):
        self.inverted_index = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.total_docs = 0

    def build_index(self, df, clean_text_column, id_column):
        self.inverted_index = {}
        self.doc_lengths = {}
        self.total_docs = len(df)
        total_words = 0

        for _, row in df.iterrows():
            doc_id = row[id_column]
            tokens = row[clean_text_column].split()
            self.doc_lengths[doc_id] = len(tokens)
            total_words += len(tokens)

            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = self.inverted_index[token].get(doc_id, 0) + 1

        if self.total_docs > 0:
            self.avg_doc_length = total_words / self.total_docs

    def get_documents(self, term):
        return self.inverted_index.get(term, {})
