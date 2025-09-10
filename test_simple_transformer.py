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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ----------------------------
# 4️⃣ Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x




# ----------------------------
# 4️⃣ Lightning Module for Transformer
# ----------------------------
class TransformerLM(pl.LightningModule):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_seq_len=35, dropout=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # MultiheadAttention expects bool mask

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        mask = self.create_causal_mask(seq_len, x.device)

        x = self.embedding(x) * math.sqrt(self.hparams.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln(x)
        logits = self.head(x)  # (batch_size, seq_len, vocab_size)
        return logits


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



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
    seq_len = 35
    batch_size = 256
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=7)

    # Model + Trainer
    model = TransformerLM(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=0.25, max_epochs=20, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
