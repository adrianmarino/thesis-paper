import requests
import json
from bunch import Bunch
from .query import OllamaQueryResult


class OllamaApiClient:

    def __init__(
        self,
        host  = 'localhost:11434',
    ):
        self.base_url = f'http://{host}/api'


    def query(self, msg, model):
        response = requests.post(
            f'{self.base_url}/generate',
            json   = { 'model': model, 'prompt': msg },
            stream = True
        )

        message = ''
        if response.status_code == 200:
            for json_obj in response.iter_lines():
                if json_obj:
                    decoded_json = Bunch(json.loads(json_obj))

                    if decoded_json.done:
                        metadata = decoded_json
                    else:
                        message += decoded_json.response

            return OllamaQueryResult(msg, message, metadata)

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
