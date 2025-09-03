import pickle
from typing import List, Tuple, Optional
from collections import Counter
import re

class WordTokenizer:
    """Word-level tokenizer for WikiText-103"""

    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = 50000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'([.!?;,:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()

    def build_vocab(self, texts: List[str]) -> None:
        if not texts:
            raise ValueError("No texts provided to build vocabulary.")
        for text in texts:
            self.word_counts.update(self._tokenize_text(text))

        most_common = [word for word, count in self.word_counts.most_common()
                       if count >= self.min_freq]
        vocab_words = most_common[:self.vocab_size - len(self.SPECIAL_TOKENS)]

        # Add special tokens
        self.word2idx = {token: i for i, token in enumerate(self.SPECIAL_TOKENS)}
        self.idx2word = {i: token for i, token in enumerate(self.SPECIAL_TOKENS)}

        # Add vocab words
        for i, word in enumerate(vocab_words):
            idx = i + len(self.SPECIAL_TOKENS)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built with {len(self.word2idx)} tokens.")

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize_text(text)
        return [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        return ' '.join([self.idx2word.get(idx, self.UNK_TOKEN) for idx in token_ids])

    def save(self, filepath: str) -> None:
        """Save the full tokenizer object with pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Tokenizer saved to {filepath}")



    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        print(f"Tokenizer loaded from {filepath}")

    @property
    def pad_token_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.word2idx[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

