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

class CBR_RNN(nn.Module):
    def __init__(self, ntoken, ninp=512, nhid=512, nheads=1, seq_len=128, compressed_dim=1, dropout=0.5, learning_rate=1e-3,
                 criterion='cross_entropy', optimizer_type='adam', weight_decay=0.0):
        super().__init__()

        # Model components
        self.encoder = nn.Embedding(ntoken+1, ninp)
        self.nhid = nhid
        self.seq_len = seq_len
        self.compressed_dim = compressed_dim
        self.nheads = nheads
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.q = nn.Linear(ninp + nhid, nhid)
        self.q_norm = nn.LayerNorm(nhid)

        self.intermediate_h = nn.Linear(nhid*3 + ninp, nhid*4)
        self.int_norm = nn.LayerNorm(nhid*4)
        self.f_norm = nn.LayerNorm(nhid*3)
        self.final_h = nn.Linear(nhid*4, nhid*3)

        self.decoder = nn.Linear(nhid, ntoken+1)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True)

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.criterion = criterion
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name or "decoder" in name:
                    nn.init.normal_(param, 0, 0.1)
                elif "multihead_attn" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(
                        param, mode="fan_in", nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    # ----------------------
    # Cache handling
    # ----------------------

    def init_cache(self, observation):
        bsz = observation.size(0) #if len(observation.size()) > 1 else 1
        hidden = torch.zeros(self.compressed_dim, bsz,
                             self.nhid)
        key_cache = torch.zeros(bsz, self.compressed_dim,
                                self.nhid)
        value_cache = torch.zeros(
            bsz, self.compressed_dim, self.nhid)
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_i, value_i, hidden_i):
        # Ensure old cache has NO graph connections
        hidden_detached = hidden.detach() if hidden.requires_grad else hidden
        key_detached = key_cache.detach() if key_cache.requires_grad else key_cache
        value_detached = value_cache.detach() if value_cache.requires_grad else value_cache

        # Now safe to concatenate
        hidden = torch.cat((hidden_detached, hidden_i.unsqueeze(0)), dim=0)
        key_cache = torch.cat((key_detached, key_i.unsqueeze(1)), dim=1)
        value_cache = torch.cat((value_detached, value_i.unsqueeze(1)), dim=1)
        return key_cache, value_cache, hidden


    def compress_cache(self, hidden, key_cache, value_cache):
        # Keep only the last N tokens (most recent)
        if hidden.size(0) > self.compressed_dim:
            hidden_compressed = hidden[-self.compressed_dim:]
            key_compressed = key_cache[:, -self.compressed_dim:]
            value_compressed = value_cache[:, -self.compressed_dim:]
        else:
            hidden_compressed = hidden
            key_compressed = key_cache
            value_compressed = value_cache

        return hidden_compressed, key_compressed, value_compressed


    # ----------------------
    # Forward
    # ----------------------
    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        q = self.drop(self.tanh(self.q_norm(self.q(combined))))
        return q.unsqueeze(1)

    def intermediate_layers(self, i, emb, query, attn, hidden):
        inter_input = torch.cat((emb[:,i,:], query, attn, hidden[-1]), -1)
        inter = self.drop(self.tanh(self.int_norm(
            self.intermediate_h(inter_input))))
        final = self.drop(self.tanh(self.f_norm(self.final_h(inter))))
        k_i, v_i, h_i = final.split(self.nhid, dim=-1)
        return k_i, v_i, h_i

    def forward(self, observation, initial_cache=None):
        seq_len = observation.size(1)
        hidden, key_cache, value_cache = initial_cache if initial_cache else self.init_cache(
            observation)
        emb = self.drop(self.encoder(observation))
        for i in range(seq_len):
            query = self.get_query(emb[:,i,:], hidden)
            attn_out, _ = self.multihead_attn(
                query, key_cache, value_cache, need_weights=False)
            attn_out = attn_out.squeeze(1)
            query = query.squeeze(1)
            k_i, v_i, h_i = self.intermediate_layers(
                i, emb, query, attn_out, hidden)
            key_cache, value_cache, hidden = self.update_cache(
                key_cache, value_cache, hidden, k_i, v_i, h_i)
        decoded = self.decoder(hidden[-self.seq_len:])#.transpose(0, 1)
        cache = self.compress_cache(hidden, key_cache, value_cache)

        return decoded, cache

  

# ----------------------------
# 4️⃣ Lightning Module for Transformer
# ----------------------------
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = CBR_RNN(ntoken=vocab_size)
        self.epoch_cache = None
        
    def _shared_step(self, batch, stage):
        data, targets = batch
        if self.epoch_cache is None:
            self.epoch_cache = self.model.init_cache(data)

        output, new_cache = self.model.forward(data, initial_cache=self.epoch_cache)

        if new_cache is not None:
            self.epoch_cache = tuple(c.detach().clone() for c in new_cache)

        output_flat, targets_flat = output.reshape(
            -1, output.size(-1)), targets.reshape(-1)
        loss = F.cross_entropy(output_flat, targets_flat)
        ppl = torch.exp(loss)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(
        batch, "train")

    def validation_step(
        self, batch, batch_idx): return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx): return self._shared_step(
        batch, "test")

    def on_train_epoch_end(self):
        self.epoch_cache = None
    
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
    trainer = pl.Trainer(gradient_clip_val=0.25, max_epochs=20, accelerator="gpu", devices=1, precision='bf16-mixed')
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
