import ollama
from rest.ollama import OllamaApiClient
import numpy as np

class OllamaSentenceEmbeddingService:
  def __init__(self, model,  host  = 'localhost:11434'):
    self._model = model
    # self._client = OllamaApiClient(host)

  def generate(self, texts):
    # return np.array(self._client.embeddings(self._model, texts))
    # return np.array([self._client.embedding(self._model, text) for text in texts])
    return np.array([ollama.embeddings(model=self._model, prompt=text)['embedding'] for text in texts])
