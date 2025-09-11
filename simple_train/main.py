import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
from pathlib import Path
from collections import Counter
from cbr_lightning.attention import MultiheadAttention
import math
from tokenizer_and_dataset import WordTokenizer, WikiTextDataset
from lightning_modules import CBR_RNN, Transformer, LSTM
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Universal model training script')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to config data dir', default = "cbr_lightning/wikitext-103-raw")
    parser.add_argument('--vocab_size', type=int, required=True,
                        help='Vocabulary size', defualt = 50000)
    parser.add_argument('--seq_len', type=int, required=True,
                        default = 128)
    parser.add_argument('--batch_size', type=int, required=True,
                        default = 256)
    parser.add_argument('--epochs', type=int, required=True, default = 80)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--model', type=str, required=True)
    
    args = parser.parse_args()
    
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium') 

    data_dir = args.data_dir
    raw = datasets.load_from_disk(data_dir)
    train_dataset = raw['train']
    val_dataset = raw['validation']
    test_dataset = raw['test']

    # Build tokenizer on train + val + test
    all_texts = list(train_dataset['text'])+ list(val_dataset['text'])+ list(test_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=args.vocab_size)

    # Datasets + Dataloaders
    seq_len = args.seq_len
    batch_size = args.batch_size
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=7, drop_last=True)
    
    
    model = args.model(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=args.grad_clip, max_epochs=args.epochs, accelerator="gpu", devices=1, precision='bf16-mixed')
    trainer.fit(model, train_loader, val_loader)
