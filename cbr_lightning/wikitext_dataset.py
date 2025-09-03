import re
import pickle
from typing import List, Tuple, Optional
from collections import Counter
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets


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
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
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


class WikiTextDataset(Dataset):
    """Dataset for WikiText-103, returning (input, target) sequences"""

    def __init__(self, texts: List[str], tokenizer: WordTokenizer, sequence_length: int = 35, stride: int = 1):
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []

        for text in texts:
            if not text.strip():
                continue
            token_ids = tokenizer.encode(text)
            for i in range(0, len(token_ids) - sequence_length, stride):
                seq = token_ids[i:i + sequence_length + 1]
                if len(seq) == sequence_length + 1:
                    self.sequences.append(seq)

        if not self.sequences:
            raise ValueError("No sequences generated from texts.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn_cbr(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    input_batch = torch.stack(inputs).transpose(0, 1)
    target_batch = torch.stack(targets).transpose(0, 1)
    return input_batch, target_batch


class WikiTextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WikiText-103"""

    def __init__(self, data_path: str, tokenizer_path: Optional[str] = None,
                 vocab_size: int = 50000, min_freq: int = 2, sequence_length: int = 35,
                 batch_size: int = 32, num_workers: int = 4, stride: int = 1):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path {self.data_path} not found.")
        print(f"Using dataset from {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        wikitext = datasets.load_from_disk(self.data_path)

        # Load or build tokenizer
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            self.tokenizer = WordTokenizer()
            self.tokenizer.load(self.tokenizer_path)
        else:
            self.tokenizer = WordTokenizer(vocab_size=self.vocab_size, min_freq=self.min_freq)
            train_texts = [t for t in wikitext['train']['text'] if t.strip()]
            self.tokenizer.build_vocab(train_texts)
            if self.tokenizer_path:
                self.tokenizer.save(self.tokenizer_path)

        if stage in ("fit", None):
            self.train_dataset = WikiTextDataset([t for t in wikitext['train']['text'] if t.strip()],
                                                 self.tokenizer, self.sequence_length, self.stride)
            self.val_dataset = WikiTextDataset([t for t in wikitext['validation']['text'] if t.strip()],
                                               self.tokenizer, self.sequence_length, self.stride)
        if stage in ("test", None):
            self.test_dataset = WikiTextDataset([t for t in wikitext['test']['text'] if t.strip()],
                                                self.tokenizer, self.sequence_length, self.stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=collate_fn_cbr, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=collate_fn_cbr, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=collate_fn_cbr, drop_last=True)


def run_sanity_checks(dataset: WikiTextDataset, tokenizer: WordTokenizer, num_samples: int = 5, stride: int = 1) -> str:
    """Perform sanity checks and return a string with the results."""
    results = []

    # 1. Tokenization checks
    for seq in random.sample(dataset.sequences, min(num_samples, len(dataset))):
        if not all(isinstance(tok, int) for tok in seq):
            results.append("Tokenization check failed: non-integer token found.")
        if not all(0 <= tok < tokenizer.vocab_size for tok in seq):
            results.append(f"Tokenization check failed: token out of range in sequence {seq}.")
        decoded = tokenizer.decode(seq)
        results.append(f"Decoded sequence: {decoded}")

    # 2. Input/target alignment
    for input_ids, target_ids in random.sample([dataset[i] for i in range(len(dataset))],
                                               min(num_samples, len(dataset))):
        if input_ids.shape != target_ids.shape:
            results.append("Input/target length mismatch.")
        if not torch.all(input_ids[1:] == target_ids[:-1]):
            results.append("Input/target alignment check failed.")

    # 3. Data integrity
    if any(len(seq) == 0 for seq in dataset.sequences):
        results.append("Empty sequence found in dataset.")

    # 4. Overlap correctness
    if stride > 1:
        for seq1, seq2 in zip(dataset.sequences[:-1], dataset.sequences[1:]):
            expected_overlap = dataset.sequence_length + 1 - stride
            actual_overlap = sum(a == b for a, b in zip(seq1[-expected_overlap:], seq2[:expected_overlap]))
            if actual_overlap != expected_overlap:
                results.append("Overlap check failed.")

    # 5. Dataset statistics
    all_tokens = [tok for seq in dataset.sequences for tok in seq]
    results.append(f"Total sequences: {len(dataset)}")
    results.append(f"Min token ID: {min(all_tokens)}, Max token ID: {max(all_tokens)}, Mean token ID: {sum(all_tokens)/len(all_tokens):.2f}")
    results.append(f"Sequence length (all sequences should match): {dataset.sequence_length + 1}")

    return "\n".join(results)


if __name__ == "__main__":
    data_path = "cbr_lightning/wikitext-103-local"  # replace with your local dataset path
    tokenizer_path = "./tokenizer.pkl"

    # Initialize data module
    dm = WikiTextDataModule(data_path=data_path, tokenizer_path=tokenizer_path)
    dm.prepare_data()
    dm.setup("fit")

    # Run sanity checks on training dataset
    sanity_results = run_sanity_checks(dm.train_dataset, dm.tokenizer, num_samples=10, stride=dm.stride)

    # Save results
    with open("sanity_check.txt", "w") as f:
        f.write(sanity_results)

    print("Sanity checks completed. Results saved to sanity_check.txt")
