import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
import math
import numpy as np
from collections import Counter
from entire_transformer import create_transformer_model# Assume CueBasedRNNModel is imported from your module
# from your_module import CueBasedRNNModel

class WordTokenizer:
    """Simple word-level tokenizer for WikiText"""
    def __init__(self, list_of_texts, vocab_size=50000):
        tokens = []
        for text in list_of_texts:
            tokens.extend(text.split())
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)  # reserve for special tokens
        self.itos = ["<unk>"] + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(tok, 0) for tok in text.split()]  # 0 = <unk>

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])


class WikiTextDataset(Dataset):
    """Dataset for WikiText that produces sequences in the format expected by CueBasedRNNModel"""
    def __init__(self, dataset, tokenizer, seq_len=35):
        self.seq_len = seq_len
        text = " ".join(list(dataset["text"]))
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.data[i:i+self.seq_len]
        target = self.data[i+1:i+self.seq_len+1]
        return seq, target



def train_cbr_model():
    """
    Training script for the CueBasedRNNModel
    """
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load WikiText-103 dataset
    data_dir = "cbr_lightning/wikitext-103-raw"  # Adjust path as needed
    raw = datasets.load_from_disk(data_dir)
    train_dataset = raw['train']
    val_dataset = raw['validation'] 
    test_dataset = raw['test']
    
    # Build tokenizer
    all_texts = list(train_dataset['text']) + list(val_dataset['text']) + list(test_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=50000)
    
    # Create datasets
    seq_len = 35  # Match reference implementation
    batch_size = 256
    
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    test_ds = WikiTextDataset(test_dataset, tokenizer, seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True
    )
    
    # Initialize model
    model = create_transformer_model(tokenizer)
    
    # if hasattr(torch, 'compile'):
    #     print("Compiling model...")
    #     model.model = torch.compile(model.model, mode='default')
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=5,
        gradient_clip_val=0.25,  # Match reference implementation
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=500,
        val_check_interval=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Allow non-deterministic ops for speed
        benchmark=True, 
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    

    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_cbr_model()