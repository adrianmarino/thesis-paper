import requests
import json
from bunch import Bunch
from .query import OllamaQuery


class OllamaApiClient:

    def __init__(
        self,
        host  = 'localhost:11434',
        model = 'llama2'
    ):
        self.base_url = f'http://{host}/api'
        self.model    = model


    def query(self, msg):
        response = requests.post(
            f'{self.base_url}/generate',
            json   = { 'model': self.model, 'prompt': msg },
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

            return OllamaQuery(msg, message, metadata)

        else:
            raise Exception(f"REST Api Respond with '{response.status_code}' status code. Detail: '{response.text}'")
