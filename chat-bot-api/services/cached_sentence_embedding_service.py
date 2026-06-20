import numpy as np
import logging

class CachedSentenceEmbeddingService:
    def __init__(self, sentence_emb_service, cache=None):
        self._service = sentence_emb_service
        self._cache = cache if cache is not None else {}

    def generate(self, texts):
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        # 1. Identify which texts are already in the cache
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        # 2. Generate only the missing embeddings in a single batch call
        if missing_texts:
            logging.info(f"Cache MISS for {len(missing_texts)} sentence embeddings. Generating...")
            new_embeddings = self._service.generate(missing_texts)
            
            # 3. Store the newly generated embeddings in the cache and fill the results
            for text, emb, original_idx in zip(missing_texts, new_embeddings, missing_indices):
                self._cache[text] = emb
                results[original_idx] = emb
        else:
            logging.info(f"Cache HIT for all {len(texts)} sentence embeddings!")

        return np.array(results)
