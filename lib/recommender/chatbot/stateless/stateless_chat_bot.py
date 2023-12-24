from model import OllamaChainBuilder
from .chat_bot_response import ChatBotResponse

class StatelessChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser
    ):
        self._chain           = OllamaChainBuilder.default(model, prompt)
        self._output_parser   = output_parser
        self._params_resolver = params_resolver

    def send(self, request, history):
        message = self._params_resolver.resolve(
            request      = request,
            chat_history = history
        )

        response = self._chain.invoke(message)

        metadata = self._output_parser.parse(response)

        return ChatBotResponse(response, metadata)