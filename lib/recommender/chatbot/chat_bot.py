from model import OllamaChainBuilder
from .chat_bot_request import ChatBotRequest
from .chat_bot_history import ChatBotHistory, ChatBotHistoryEntry


class ChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser,
        chat_bot_prompt
    ):
        self._chain           = OllamaChainBuilder.default(model, prompt)
        self._output_parser   = output_parser
        self._chat_bot_prompt = chat_bot_prompt
        self._params_resolver = params_resolver
        self.memory           = []
        self.history          = []


    def _build_params(self, request):
        return self._params_resolver.resolve(
            request      = request,
            chat_history = self.memory
        )


    def _save_response(self, request, response):
        self.history.append(ChatBotHistoryEntry(
            request,
            response,
            self._output_parser.parse(response)
        ))
        self.memory.append((request, response))


    def _create_request(self):
        return ChatBotRequest(
            request_prompt = self._chat_bot_prompt,
            first          = len(self.history) == 0
        )


    def start(self):
        while True:
            request = self._create_request().content
            if request == None:
                break

            message = self._build_params(request)

            print('\n')
            response = self._chain.invoke(self._build_params(request))
            print('\n')

            self._save_response(request, response)

        return ChatBotHistory(self.history)