import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
from pathlib import Path
from collections import Counter
import math

# ----------------------------
# 1️⃣ Tokenizer
# ----------------------------
class WordTokenizer:
    def __init__(self, list_of_texts, vocab_size=50000):
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
# 3️⃣ Custom Multihead Attention with Causal Masking
# ----------------------------
class CausalMultiheadAttention(nn.Module):
    def __init__(self, nhid, nheads=1, dropout=0.1):
        super().__init__()
        self.nhid = nhid
        self.nheads = nheads
        self.head_dim = nhid // nheads
        
        assert nhid % nheads == 0, "nhid must be divisible by nheads"
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(nhid, nhid)
        self.k_proj = nn.Linear(nhid, nhid)
        self.v_proj = nn.Linear(nhid, nhid)
        self.out_proj = nn.Linear(nhid, nhid)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_cache, value_cache, position_i):
        """
        Args:
            query: [batch, nhid] - query for current position
            key_cache: [batch, cache_len, nhid] - cached keys
            value_cache: [batch, cache_len, nhid] - cached values  
            position_i: int - current position (for causal masking)
        """
        batch_size = query.size(0)
        cache_len = key_cache.size(1)
        
        # Project current query
        q = self.q_proj(query)  # [batch, nhid]
        
        # Project cached keys and values
        k = self.k_proj(key_cache)  # [batch, cache_len, nhid]
        v = self.v_proj(value_cache)  # [batch, cache_len, nhid]
        
        # Reshape for multihead attention
        q = q.view(batch_size, self.nheads, self.head_dim)  # [batch, nheads, head_dim]
        k = k.view(batch_size, cache_len, self.nheads, self.head_dim).transpose(1, 2)  # [batch, nheads, cache_len, head_dim]
        v = v.view(batch_size, cache_len, self.nheads, self.head_dim).transpose(1, 2)  # [batch, nheads, cache_len, head_dim]
        
        # Compute attention scores
        q_expanded = q.unsqueeze(-1)  # [batch, nheads, head_dim, 1]
        attn_scores = torch.matmul(k, q_expanded).squeeze(-1)  # [batch, nheads, cache_len]
        
        # Apply causal masking - can only attend to positions 0 through position_i
        causal_mask = torch.zeros(cache_len, device=query.device)
        if position_i + 1 < cache_len:
            causal_mask[position_i + 1:] = float('-inf')
        
        # Add causal mask to all heads
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # [batch, nheads, cache_len]
            
        # Scale and apply softmax
        attn_scores = attn_scores * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, nheads, cache_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, nheads, cache_len, 1]
        attn_out = (attn_weights_expanded * v).sum(dim=2)  # [batch, nheads, head_dim]
        
        # Concatenate heads and project
        attn_out = attn_out.view(batch_size, self.nhid)  # [batch, nhid]
        attn_out = self.out_proj(attn_out)
        
        return attn_out


# ----------------------------
# 4️⃣ Fixed Transformer Model
# ----------------------------
class Custom_model(nn.Module):
    def __init__(self, ntoken, ninp=512, nhid=512, nheads=1, seq_len=35, dropout=0.5):
        super().__init__()

        # Model components
        self.encoder = nn.Embedding(ntoken, ninp)  # Fixed: removed +1
        self.nhid = nhid
        self.seq_len = seq_len
        self.nheads = nheads
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.q = nn.Linear(ninp + nhid, nhid)
        self.q_norm = nn.LayerNorm(nhid)

        self.intermediate_h = nn.Linear(nhid*3 + ninp, nhid*4)
        self.int_norm = nn.LayerNorm(nhid*4)
        self.f_norm = nn.LayerNorm(nhid*3)
        self.final_h = nn.Linear(nhid*4, nhid*3)

        self.decoder = nn.Linear(nhid, ntoken)  # Fixed: removed +1
        
        # Use custom causal attention instead of PyTorch's
        self.multihead_attn = CausalMultiheadAttention(nhid, nheads, dropout)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name or "decoder" in name:
                    nn.init.normal_(param, 0, 0.1)
                else:
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation):
        device = observation.device
        bsz = observation.size(0)
        hidden = torch.zeros(1, bsz, self.nhid, device=device)
        key_cache = torch.zeros(bsz, 1, self.nhid, device=device)
        value_cache = torch.zeros(bsz, 1, self.nhid, device=device)
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_i, value_i, hidden_i):
        # Detach old cache to prevent gradient issues
        hidden_detached = hidden.detach()
        key_detached = key_cache.detach()
        value_detached = value_cache.detach()

        # Concatenate new states
        hidden = torch.cat((hidden_detached, hidden_i.unsqueeze(0)), dim=0)
        key_cache = torch.cat((key_detached, key_i.unsqueeze(1)), dim=1)
        value_cache = torch.cat((value_detached, value_i.unsqueeze(1)), dim=1)
        return key_cache, value_cache, hidden

    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        q = self.drop(self.tanh(self.q_norm(self.q(combined))))
        return q

    def intermediate_layers(self, i, emb, query, attn, hidden):
        inter_input = torch.cat((emb[:,i,:], query, attn, hidden[-1]), -1)
        inter = self.drop(self.tanh(self.int_norm(self.intermediate_h(inter_input))))
        final = self.drop(self.tanh(self.f_norm(self.final_h(inter))))
        k_i, v_i, h_i = final.split(self.nhid, dim=-1)
        return k_i, v_i, h_i

    def forward(self, observation, initial_cache=None):
        seq_len = observation.size(1)
        hidden, key_cache, value_cache = initial_cache if initial_cache else self.init_cache(observation)
        emb = self.drop(self.encoder(observation))
        
        for i in range(seq_len):
            query = self.get_query(emb[:,i,:], hidden)
            
            # Use custom causal attention with position information
            attn_out = self.multihead_attn(query, key_cache, value_cache, position_i=i)
            
            k_i, v_i, h_i = self.intermediate_layers(i, emb, query, attn_out, hidden)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, k_i, v_i, h_i)
        
        # Use reference model approach: skip initial hidden state
        output = hidden[1:]  # Skip position 0, use [1, 2, ..., seq_len]
        decoded = self.decoder(output.transpose(0, 1))  # Convert to [batch, seq_len, vocab_size]
        
        return decoded


# ----------------------------
# 5️⃣ Lightning Module for Transformer
# ----------------------------
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Custom_model(ntoken=vocab_size)
        
    def _shared_step(self, batch, stage):
        data, targets = batch
        initial_cache = self.model.init_cache(data)
        output = self.model.forward(data, initial_cache=initial_cache)

        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(output_flat, targets_flat)
        ppl = torch.exp(loss)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=1.0, 
            momentum=0.9, 
            nesterov=True
        )
        return optimizer


# ----------------------------
# 6️⃣ Training
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
    all_texts = list(train_dataset['text']) + list(val_dataset['text']) + list(test_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=50000)

    # Datasets + Dataloaders
    seq_len = 35
    batch_size = 256
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=7, drop_last=True)

    # Model + Trainer
    model = LanguageModel(tokenizer.vocab_size)
    trainer = pl.Trainer(gradient_clip_val=1.0, max_epochs=5, accelerator="gpu", devices=1, precision='bf16-mixed')
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()