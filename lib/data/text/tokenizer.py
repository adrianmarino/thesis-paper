from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


def is_stop_word(token):
    return not token.is_stop and not token.is_punct and not token.like_num


class TokenizerService:
    def __init__(self, nlp = English()):
        self.tokenizer = nlp.tokenizer

    def __call__(self, text):
        return [token.text for token in self.tokenizer(text) if is_stop_word(token)]
