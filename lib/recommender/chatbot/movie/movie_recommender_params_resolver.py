from ..params_resolver import ParamsResolver

class MovieRecommenderParamsResolver(ParamsResolver):
    def __call__(self, **kargs):
        import util as ut
        return {
            'request'      : kargs["request"] if 'request' in kargs else '',
            'user_profile' : ut.to_json(kargs['user_profile']) if 'user_profile' in kargs else '',
            'user_history' : ut.to_json(kargs['user_history']) if 'user_history' in kargs else '',
            'candidates'   : ut.to_json(kargs['candidates']) if 'candidates' in kargs else '',
            'limit'        : kargs['limit'] if 'limit' in kargs else 5,
        }
