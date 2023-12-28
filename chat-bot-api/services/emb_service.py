
from sentence_transformers import SentenceTransformer


class EmbService:
  def __init__(self, model_name):
    self.model = SentenceTransformer(model_name)

  def embeddings(self, texts):
    return self.model.encode(texts)
