from bunch import Bunch


class OllamaChatParamsResolver:
    def resolve(self, **kargs):
        return Bunch(
            user_profile = kargs['user_profile'] if 'user_profile' in kargs else 'No existe información de su perfíl',
            user_history = kargs['user_history'] if 'user_history' in kargs else 'Aun no se registraron películas vista',
            candidates   = kargs['candidates'] if 'candidates' in kargs else 'Aun no se dispone de películas cantidatas',
            limit        = kargs['limit'] if 'limit' in kargs else 5,
            query        = f'Usuario: {kargs["query"]}.' if 'query' in kargs else '¿Me recomendarias una película?',
        )