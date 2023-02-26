import pickle


class PickleUtils:
    @staticmethod
    def save(obj, path):
        with open(f'{path}.pkl', 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load(path):
        with open(f'{path}.pkl', 'rb') as file:
            return pickle.load(file)