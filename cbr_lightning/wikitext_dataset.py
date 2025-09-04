import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import datasets
import numpy as np
import random
import pytorch_lightning as pl
from collections import Counter
# from word_tokenizer import WordTokenizer


class TokenizedTextDataset(Dataset):
    """Dataset for next-token prediction with lazy sequence slicing (no padding)."""
    
    def __init__(self, tokenized_ids, sequence_length=128, stride=64, tokenizer=None, sanity_checks=True):
        self.tokenized_ids = np.array(tokenized_ids, dtype=np.int32)  # fast slicing
        self.sequence_length = sequence_length
        self.stride = stride
        self.tokenizer = tokenizer

        # number of sequences (skip last partial sequence)
        total_tokens = len(self.tokenized_ids)
        self.num_sequences = max(0, (total_tokens - sequence_length) // stride)

        if sanity_checks:
            self.run_sanity_checks()

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sequence_length + 1
        seq = self.tokenized_ids[start:end]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

    def run_sanity_checks(self, num_samples=5):
        print("Running dataset sanity checks...")
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        for _ in range(min(num_samples, len(self))):
            idx = random.randint(0, len(self)-1)
            x, y = self[idx]
            assert x.dtype == torch.long and y.dtype == torch.long
            if vocab_size is not None:
                assert all(0 <= t < vocab_size for t in x.tolist() + y.tolist())
            assert (x[1:] == y[:-1]).all()
            if self.tokenizer is not None:
                decoded = self.tokenizer.decode(torch.cat([x, y[-1:]]).tolist())
                print(f"✔ Sample decoded: {decoded[:100]}...")
        print("All dataset sanity checks passed ✅")

    


    def diagnostic_report(self, num_samples: int = 5) -> None:
        print("\n--- DATASET DIAGNOSTIC REPORT ---")

        # Gather all tokens from inputs
        all_tokens = torch.cat([self[i][0] for i in range(len(self))]).tolist()
        total = len(all_tokens)
        unk_count = all_tokens.count(self.tokenizer.unk_token_id)
        print(f"Total tokens: {total}")
        print(f"OOV rate: {unk_count/total:.2%}")

        # Special tokens count
        for special, tid in [
            ("PAD", self.tokenizer.pad_token_id),
            ("UNK", self.tokenizer.unk_token_id),
            ("BOS", self.tokenizer.bos_token_id),
            ("EOS", self.tokenizer.eos_token_id),
        ]:
            print(f"{special} count: {all_tokens.count(tid)}")

        # Top 10 frequent tokens
        top = Counter(all_tokens).most_common(10)
        print("Top 10 tokens:", [(self.tokenizer.idx2word.get(t, '<unk>'), c) for t, c in top])

        # Sequence length stats
        lengths = [len(self[i][0]) for i in range(len(self))]
        print(f"Sequence length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.2f}")

        # Alignment check
        print("\nSample input/target pairs (decoded):")
        for i in torch.randint(0, len(self), (num_samples,)):
            x, y = self[i]
            print("Input :", self.tokenizer.decode(x.tolist()[:20]))
            print("Target:", self.tokenizer.decode(y.tolist()[:20]))
            print("---")

        # Integrity check
        bad = [tok for tok in all_tokens if tok < 0 or tok >= self.tokenizer.vocab_size]
        if bad:
            print(f"❌ Found {len(bad)} invalid token IDs!")
        else:
            print("✔ All token IDs are within valid range.")

        print("--- END OF REPORT ---\n")

class WikiTextDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for tokenized Wikitext."""

    def __init__(self, tokenized_path: str, tokenizer, sequence_length=128, stride=64,
                 batch_size=32, num_workers=4, pin_memory=True):
        super().__init__()
        self.tokenized_path = Path(tokenized_path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        tokenized_ds = datasets.load_from_disk(self.tokenized_path)

        def flatten_split(split_name):
            return np.concatenate([np.array(seq, dtype=np.int32) for seq in tokenized_ds[split_name]['input_ids']])

        if stage in ('fit', None):
            train_ids = flatten_split('train')
            val_ids = flatten_split('validation')
            self.train_dataset = TokenizedTextDataset(train_ids, self.sequence_length, self.stride, self.tokenizer)
            self.val_dataset = TokenizedTextDataset(val_ids, self.sequence_length, self.stride, self.tokenizer, sanity_checks=False)

        if stage in ('test', None):
            test_ids = flatten_split('test')
            self.test_dataset = TokenizedTextDataset(test_ids, self.sequence_length, self.stride, self.tokenizer, sanity_checks=False)

    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            prefetch_factor=4
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def run_sanity_checks(self) -> None:
        print("Running DataModule sanity checks...\n")

        for name, dataset in [
            ("TRAIN", self.train_dataset),
            ("VAL", self.val_dataset),
            ("TEST", self.test_dataset),
        ]:
            if dataset is not None:
                print(f"--- {name} DATASET ---")
                dataset.run_sanity_checks()
                dataset.diagnostic_report(num_samples=3)  # deeper check
        print("All DataModule sanity checks passed ✅")


if __name__ == "__main__":
    import pickle

    tokenized_dataset_path = "cbr_lightning/wikitext-103-tokenized"
    tokenizer_path = "./tokenizer.pkl"
    

    from word_tok import WordTokenizer  # your local tokenizer.py

    tokenizer = WordTokenizer()
    tokenizer.load(tokenizer_path)


    dm = WikiTextDataModule(tokenized_path=tokenized_dataset_path, tokenizer=tokenizer)
    dm.setup('fit')
    dm.run_sanity_checks()