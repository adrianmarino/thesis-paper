import json


def printable(cls):
    def __str__(self): return str(self._state() if hasattr(self, '_state') else self)
    cls.__str__ = __str__


    def __repr__(self): return self.to_json()
    cls.__repr__ = __repr__


    def to_json(self):
        return json.dumps(
            self._state() if hasattr(self, '_state') else self,
            default   = lambda o: o.__dict__,
            sort_keys = True,
            indent    = 4
        )
    cls.to_json = to_json

    return cls