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
# ----------------------------
# 1️⃣ Tokenizer
# ----------------------------
class WordTokenizer:
    def __init__(self, list_of_texts, vocab_size=50000):
        tokens = []
        tokens = []
        for text in list_of_texts:
            tokens.extend(text.split())
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)  # reserve 0: <pad>, 1: <unk>
        self.itos = ["<unk>"] + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(tok, 1) for tok in text.split()]  # 1 = <unk>

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])


# ----------------------------
# 2️⃣ Dataset
# ----------------------------
class WikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len=35):
        self.seq_len = seq_len
        text = " ".join(list(dataset["text"]))
        # Encode text to integer token IDs
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.data[i:i+self.seq_len]
        target = self.data[i+1:i+self.seq_len+1]
        return seq, target


# ----------------------------
# 3️⃣ Transformer Model
# ----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.pos_encoding(x)
        # Causal mask for language modeling
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits  # shape: (batch, seq_len, vocab_size)


# ----------------------------
# 4️⃣ Lightning Module for Transformer
# ----------------------------
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = TransformerLM(vocab_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0)  # peak LR

        def lr_lambda(step):
            warmup_steps = 4000
            d_model = 512
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

# ----------------------------
# 5️⃣ Training
# ----------------------------
def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium') 

    data_dir = "cbr_lightning/wikitext-103-raw"
    raw = datasets.load_from_disk(data_dir)
    train_dataset = raw['train']
    val_dataset = raw['validation']
    test_dataset = raw['test']

    # Build tokenizer on train + val + test
    all_texts = list(train_dataset['text'])+ list(val_dataset['text'])+ list(test_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=50000)

    # Datasets + Dataloaders
    seq_len = 128
    batch_size = 512
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=7)

    # Model + Trainer
    model = LanguageModel(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=0.25, max_epochs=20, accelerator="gpu", devices=1, precision='transformer-engine-float16')
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
