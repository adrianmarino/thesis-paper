class OllamaQueryResult:
    def __init__(self, query, response, metadata):
        self.query    = query
        self.response = response
        self.metadata = metadata

    def __repr__(self): return f'Query:\n{self.query}\n\nResponse:{self.response}'
