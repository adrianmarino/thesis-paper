import util as ut


class ChatBotHistory:
    def __init__(self, entries):
        self.entries = entries

    def __repr__(self): return ut.to_json(self, sort_keys=False)

    def __str__(self): return self.__repr__()


class ChatBotHistoryEntry:
    def __init__(self, request, response, data):
        self.request = request
        self.response = response
        self.data = data

    def __repr__(self): return ut.to_json(self, sort_keys=False)

    def __str__(self): return self.__repr__()
