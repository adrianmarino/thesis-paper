from bunch import Bunch
from ..params_resolver import ParamsResolver


class MovieRecommenderParamsResolver(ParamsResolver):
    def resolve(self, **kargs):
        return Bunch(
            user_profile = kargs['user_profile'] if 'user_profile' in kargs else 'No existe información de su perfíl',
            user_history = kargs['user_history'] if 'user_history' in kargs else 'Aun no se registraron películas vista',
            candidates   = kargs['candidates'] if 'candidates' in kargs else 'Aun no se dispone de películas cantidatas',
            limit        = kargs['limit'] if 'limit' in kargs else 5,
            request      = f'Usuario: {kargs["request"]}.' if 'request' in kargs else '¿Me recomendarias una película?',
            chat_history = kargs['memory'] if 'memory' in kargs else None
        )
