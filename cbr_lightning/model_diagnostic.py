# sanity_check_models.py
import torch
from torch.utils.data import DataLoader
import datasets
from model_lightning import CBR_RNN, Transformer, LSTM  # your models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load tokenized dataset ---
dataset_path = "cbr_lightning/wikitext-103-tokenized"
ds = datasets.load_from_disk(dataset_path)

# --- Small batch for sanity check ---
batch_size = 2
split = "train"

def collate_batch(batch):
    """Convert list of dicts to tensor batch"""
    input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    target_ids = torch.tensor([b['target_ids'] for b in batch], dtype=torch.long)
    return input_ids, target_ids

dataloader = DataLoader(ds[split], batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Get a single batch
input_ids, target_ids = next(iter(dataloader))
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

print("Input shape:", input_ids.shape)   # (batch, seq_len)
print("Target shape:", target_ids.shape) # (batch, seq_len)
print("Token range: min =", input_ids.min().item(), "max =", input_ids.max().item())

# Vocabulary size and sequence length
vocab_size = max(input_ids.max().item(), target_ids.max().item()) + 1
seq_len = input_ids.size(1)

# --- CBR_RNN ---
print("\n=== Testing CBR_RNN ===")
rnn_model = CBR_RNN(ntoken=vocab_size, ninp=256, nhid=512, nheads=4, seq_len=seq_len, compressed_dim=32)
rnn_model.to(device)
rnn_model.train()

# Forward pass (CBR_RNN expects seq_len first)
rnn_out, rnn_cache = rnn_model(input_ids.transpose(0, 1))
print("Output shape:", rnn_out.shape)  # (seq_len, batch, vocab_size)

# Backward pass
loss_rnn = torch.nn.functional.cross_entropy(
    rnn_out.reshape(-1, vocab_size),
    target_ids.transpose(0, 1).reshape(-1)
)
loss_rnn.backward()
print("CBR_RNN forward/backward pass successful ✅")

# --- Transformer ---
print("\n=== Testing Transformer ===")
transformer_model = Transformer(
    vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, max_seq_len=seq_len, dropout=0.1,
    lr=1e-3, temperature=1.0, gumbel_softmax=False
)
transformer_model.to(device)
transformer_model.train()

# Forward pass
transformer_out = transformer_model(input_ids)
print("Output shape:", transformer_out.shape)  # (seq_len, batch, vocab_size)

# Backward pass
loss_trans = torch.nn.functional.cross_entropy(
    transformer_out.reshape(-1, vocab_size),
    target_ids.transpose(0, 1).reshape(-1)
)
loss_trans.backward()
print("Transformer forward/backward pass successful ✅")

# --- LSTM ---
print("\n=== Testing LSTM ===")
lstm_model = LSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512)
lstm_model.to(device)
lstm_model.train()

# Forward pass
lstm_out, _ = lstm_model(input_ids)
print("Output shape:", lstm_out.shape)  # (batch, seq_len, vocab_size)

# Backward pass
loss_lstm = torch.nn.functional.cross_entropy(
    lstm_out.reshape(-1, vocab_size),
    target_ids.reshape(-1)
)
loss_lstm.backward()
print("LSTM forward/backward pass successful ✅")

print("\n✅ All models passed sanity check with dataset batch.")
