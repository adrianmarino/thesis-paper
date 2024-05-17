import requests
import json
from bunch import Bunch
from .query import OllamaQueryResult
from util import ParallelExecutor

class OllamaApiClient:

    def __init__(
        self,
        host  = 'localhost:11434',
    ):
        self.base_url = f'http://{host}/api'


    def query(self, msg, model):
        response = requests.post(
            f'{self.base_url}/generate',
            json   = { 'model': model, 'prompt': msg, 'stream': False},
        )

        message = ''
        if response.status_code == 200:
            body = response.json()

            answer = body.get('response', None)
            if message:
                del body['response']

            return OllamaQueryResult(
                query=msg,
                response=answer,
                metadata=body
            )

        else:
            raise Exception(f"REST Api Respond with '{response.status_code}' status code. Detail: '{response.text}'")


    def models(self):
        return [tag['name'] for tag in self.tags()]


    def tags(self):
        response = requests.get(f'{self.base_url}/tags')

        if response.status_code == 200:
            return response.json()['models']
        else:
            raise Exception(f"REST Api Respond with '{response.status_code}' status code. Detail: '{response.text}'")


    def embedding(self, model, prompt):
        response = requests.post(
            f'{self.base_url}/embeddings',
            json = { 'model': model, 'prompt': prompt}
        )

        if response.status_code == 200:
            return response.json().get('embedding', None)
        else:
            raise Exception(f"REST Api Respond with '{response.status_code}' status code. Detail: '{response.text}'")


    def _embedding_fn(self, index, model, prompt): return (index, self.embedding(model, prompt))


    def embeddings(self, model, prompts, n_processes = 24):
        result = ParallelExecutor(n_processes)(
            self._embedding_fn,
            params = [[index, model, prompt] for index, prompt in enumerate(prompts)],
            fallback_result = None
        )

        return list(dict(sorted({e[0]:e[1]  for e in result}.items())).values())

