import pickle
import logging


class Picket:
    @staticmethod
    def save(path, obj):
        with open(f'{path}', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(f'{path}', 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def try_load(path):
        try:
            with open(f'{path}', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError as e:
            logging.warn(f"Missing '{path}' file")
            return None
