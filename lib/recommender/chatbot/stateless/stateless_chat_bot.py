import importlib

ml = importlib.import_module('model')
from .chat_bot_response_factory import ChatBotResponseFactory
import logging
import sys

class StatelessChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser
    ):
        self._model  = model
        self._prompt = prompt

        self._params_resolver  = params_resolver
        template_factory       = ml.OllamaChatPromptTemplateFactory.create(prompt)
        self._chain            = template_factory | ml.OllamaModelBuilder.chat(model)
        self._response_factory = ChatBotResponseFactory(output_parser, template_factory)


    @property
    def name(self): return f'Model: {self._model}. Prompt: {self._prompt}'


    def send(self, **kargs):
        params = self._params_resolver.resolve(**kargs)

        response = self._chain.invoke(params)

        return self._response_factory.create(kargs, response)
