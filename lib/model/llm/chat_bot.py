from .chain_builder import OllamaChainBuilder


class ChatBot:
    def __init__(
        self,
        model,
        prompt,
        params_resolver,
        output_parser,
        message = '¿Que queres que te recomiende hoy?'
    ):
        self.__chain = OllamaChainBuilder.default(model, prompt)
        self.__output_parser = output_parser
        self.__message = message
        self.__params_resolver = params_resolver
        self.memory = []
        self.recommendations = []


    def _build_params(self, query):
        return self.__params_resolver.resolve(input=query, chat_history=self.memory)


    def _save_response(self, query, response):
        self.recommendations.append(self.__output_parser.parse(response))
        self.memory.append((query, response))


    def start(self):
        while True:
            query = input(f"\n{self.__message} ('\\bye' para finalizar)\n\n")
            if query == "\\bye":
                break

            if len(query.strip()) > 0:
                print('\n')
                response = self.__chain.invoke(self._build_params(query))
                self._save_response(query, response)

        return self.recommendations