from model import OllamaChainBuilder, OllamaChatPromptTemplateFactory, OllamaModelBuilder
from .chat_bot_response_factory import ChatBotResponseFactory
import logging


class StatelessChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser
    ):
        self._params_resolver  = params_resolver
        template_factory       = OllamaChatPromptTemplateFactory.create(prompt)
        self._chain            = template_factory | OllamaModelBuilder.default(model)
        self._response_factory = ChatBotResponseFactory(output_parser, template_factory)


    def send(self, **kargs):
        params = self._params_resolver.resolve(**kargs)

        response = self._chain.invoke(params)

        return self._response_factory.create(kargs, response)
