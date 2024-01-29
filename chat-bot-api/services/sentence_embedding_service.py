
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingService:
  def __init__(self, model):
    self.model = SentenceTransformer(model)

  def generate(self, texts):
    return self.model.encode(texts, show_progress_bar=False)
