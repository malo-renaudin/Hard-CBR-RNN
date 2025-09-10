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
# 3️⃣ LSTM Model
# ----------------------------
class TemperatureScheduler:
    """Simple exponential decay temperature scheduler"""

    def __init__(self, initial_temp=1.0, decay_rate=0.95, final_temp=0.1):
        self.initial_temp = initial_temp
        self.decay_rate = decay_rate
        self.final_temp = final_temp
        self.current_epoch = 0

    def step(self):
        """Update the current epoch"""
        self.current_epoch += 1

    def get_temperature(self):
        """Get current temperature using exponential decay"""
        temp = self.initial_temp * (self.decay_rate ** self.current_epoch)
        return max(temp, self.final_temp)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temperature, gumbel_softmax, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            x, x, x,  attn_mask=mask, temperature=temperature, gumbel_softmax=gumbel_softmax, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Transformer(pl.LightningModule):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, max_seq_len, dropout, temperature, gumbel_softmax, learning_rate, temp_decay_rate=0.95, temp_final=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        self.temp_decay_rate = temp_decay_rate
        self.temp_final = temp_final

        self.temp_scheduler = TemperatureScheduler(
            initial_temp=temperature,
            decay_rate=temp_decay_rate,
            final_temp=temp_final
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size+1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size+1, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(
            seq_len, seq_len, device=device), diagonal=1)
        mask = mask.float()                 # ensure float
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def forward(self, input_ids, temperature, gumbel_softmax):
        temperature = self.temperature
        gumbel_softmax = self.gumbel_softmax
        seq_len, batch_size = input_ids.size()
        input_ids = input_ids.transpose(0, 1)
        print('input_ids', input_ids.shape)
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        # x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, input_ids.device)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask, temperature=temperature,
                      gumbel_softmax=gumbel_softmax)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits



# ----------------------------
# 4️⃣ Lightning Module
# ----------------------------
# ----------------------------
# 4️⃣ Lightning Module for Transformer
# ----------------------------
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_seq_len=35, dropout=0.1,
                 temperature=1.0, gumbel_softmax=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            temperature=temperature,
            gumbel_softmax=gumbel_softmax,
            learning_rate=learning_rate
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, self.hparams.temperature, self.hparams.gumbel_softmax)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, self.hparams.temperature, self.hparams.gumbel_softmax)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
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
    model = LanguageModel(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=0.25, max_epochs=20, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
