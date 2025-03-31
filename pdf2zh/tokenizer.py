from tiktoken import get_encoding

from pdf2zh.utils.singleton import SingletonMeta


class TokenizerManager(metaclass=SingletonMeta):
    def __init__(self):
        self._tokenizer = get_encoding("cl100k_base")

    @classmethod
    def count_tokens(cls, text: str):
        instance = cls()
        return len(instance._tokenizer.encode(text))
