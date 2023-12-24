from model import OllamaChainBuilder
from .chat_bot_request import ChatBotRequestFactory
from .chat_bot_history import ChatBotHistory, ChatBotHistoryEntry


class TextChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser,
        request_prompt
    ):
        self._chain           = OllamaChainBuilder.default(model, prompt)
        self._output_parser   = output_parser
        self._params_resolver = params_resolver
        self._request_prompt  = request_prompt
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


    def start(self, user_profile):
        request_factory = ChatBotRequestFactory(
            self._request_prompt,
            user_profile,
            self.history
        )

        while True:
            request = request_factory.create().content

            if request == None:
                break

            message = self._build_params(request)

            print('\n')
            response = self._chain.invoke(self._build_params(request))
            print('\n')

            self._save_response(request, response)

        return ChatBotHistory(self.history)