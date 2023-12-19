from model.llm import OllamaChatParamsResolver, OllamaChainBuilder, MovieRecommenderDiscListOutputParser


class MovieRecommenderChatBot:
    def __init__(
        self,
        model,
        promp,
        list_size,
        message = '¿Que queres que te recomiende hoy?'
    ):
        self.__chain = OllamaChainBuilder().chat(model, promp)
        self.__output_parser = MovieRecommenderDiscListOutputParser(list_size=list_size)
        self.__message = message
        self.memory = []
        self.recommendations = []


    def _build_params(self, query):
        return OllamaChatParamsResolver().resolve(query=query, chat_history=self.memory)


    def _save_response(self, query, response):
        self.recommendations.append(self.__output_parser.parse(response.content))
        self.memory.append((query, response.content))


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