from nltk import sent_tokenize
import nltk
from sentence_transformers import CrossEncoder

class CrossEncoderDocumentRanker:
    def __init__(self, modelName="ms-marco-TinyBERT-L-2"):
        nltk.download("punkt")
        nltk.download("punkt_tab")
        self.__model = CrossEncoder(f"cross-encoder/{modelName}")

    def __call__(self, query, documents):
        return self.__model.rank(query, documents, return_documents=True)