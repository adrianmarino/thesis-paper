from datetime import datetime

import util as ut


class LogPathBuilder:
    def __init__(
            self,
            path=None,
            time_pattern='%Y-%m-%d_%H-%M-%S'
    ):
        self._path = path
        self._time_pattern = time_pattern

    def __str_now(self):
        return datetime.now().strftime(self._time_pattern)

    def build(self, filename):
        return f'{ut.mkdir(self._path)}/{self.__str_now()}-{filename}' if self._path else None
