from rest.ollama import OllamaApiClient
import logging

class ChatBotClient:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser,
        host  = 'localhost:11434'
    ):
        self._model  = model
        self._prompt = prompt

        self._params_resolver  = params_resolver
        self._template         = PromptTemplate(prompt)
        self._client           = OllamaApiClient(host)
        self._output_parser    = output_parser

    @property
    def name(self): return f'Model: {self._model}. Prompt: {self._prompt}'

    def __call__(self, **kargs):
        params = self._params_resolver(**kargs)

        query = self._template(params)

        logging.debug(f'PROMPT: {query}')

        result = self._client.query(
            msg   = self._template(params),
            model = self._model
        )

        logging.info(f"""
PROMPT:
{query}

ANSWER:
{result.response}
""")

        return self._create_result(result, params)

    def _create_result(self, result, params):
        metadata = self._output_parser.parse(result.response, lines_size=params['limit'])
        metadata['params'] = params
        metadata['prompt'] = result.query

        return ChatBotResult(result.response, metadata)

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def __call__(self, params):
        result = self.template
        for name, value in params.items():
            result = result.replace('{' + name  + '}', self._sanitize(value))
        return result

    def _sanitize(self,value):
        value = value if value else ''

        value = str(value).replace('\\n', '\n')

        if value.startswith('"'): value = value[1:]
        if value.endswith('"'): value = value[:-1]

        return value.replace('\\n', '\n')


class ChatBotResult:
  def __init__(self, content, metadata):
    self.content = content
    self.metadata = metadata

