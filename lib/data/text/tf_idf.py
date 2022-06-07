
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfGenerator:
    def __init__(self, ngram_range=(1, 1), min_df=0.0001, stop_words='english'):
        self._vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words=stop_words)
    
    def __call__(self, documents):
        return self._vectorizer.fit_transform(documents)