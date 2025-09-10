import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from collections import Counter

# ----------------------------
# 1️⃣ Tokenizer
# ----------------------------
class WordTokenizer:
    def __init__(self, files, vocab_size=50000):
        tokens = []
        for file in files:
            text = Path(file).read_text(encoding="utf-8")
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
    def __init__(self, file_path, tokenizer, seq_len=35):
        self.seq_len = seq_len
        text = Path(file_path).read_text(encoding="utf-8")
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
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid_dim=512, nlayers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hid_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.nlayers = nlayers
        self.hid_dim = hid_dim

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.hid_dim, device=device),
                weight.new_zeros(self.nlayers, batch_size, self.hid_dim, device=device))


# ----------------------------
# 4️⃣ Lightning Module
# ----------------------------
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = SimpleLSTM(vocab_size)
        self.hidden = None

    def on_train_epoch_end(self):
        self.hidden = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        device = x.device

        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.hidden = self.model.init_hidden(batch_size, device)

        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        logits, self.hidden = self.model(x, self.hidden)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        device = x.device
        hidden = self.model.init_hidden(batch_size, device)
        logits, _ = self.model(x, hidden)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=-100)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=20.0)


# ----------------------------
# 5️⃣ Training
# ----------------------------
def main():
    pl.seed_everything(42)

    data_dir = "wikitext-103-raw"
    train_file = Path(data_dir) / "train"
    val_file = Path(data_dir) / "validation"
    test_file = Path(data_dir) / "test"

    # Build tokenizer on train + val + test
    tokenizer = WordTokenizer([train_file, val_file, test_file], vocab_size=50000)

    # Datasets + Dataloaders
    seq_len = 35
    batch_size = 20
    train_dataset = WikiTextDataset(train_file, tokenizer, seq_len)
    val_dataset = WikiTextDataset(val_file, tokenizer, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model + Trainer
    model = LanguageModel(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=0.25, max_epochs=20, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
