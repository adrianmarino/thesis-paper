import logging

class CachedOllamaApiClient:
    def __init__(self, client, cache=None):
        self._client = client
        self._cache = cache if cache is not None else {}

    def invalidate(self, msg, model):
        cache_key = (model, msg)
        if cache_key in self._cache:
            logging.info(f"Invalidating cache for model '{model}'")
            del self._cache[cache_key]

    def query(self, msg, model):
        cache_key = (model, msg)
        if cache_key in self._cache:
            logging.info(f"Cache HIT for model '{model}'")
            return self._cache[cache_key]

        logging.info(f"Cache MISS for model '{model}'. Directing request to Ollama...")
        result = self._client.query(msg, model)
        self._cache[cache_key] = result
        return result

    def models(self):
        return self._client.models()

    def tags(self):
        return self._client.tags()

    def embedding(self, model, prompt):
        return self._client.embedding(model, prompt)

    def embeddings(self, model, prompts, n_processes=24):
        return self._client.embeddings(model, prompts, n_processes)
