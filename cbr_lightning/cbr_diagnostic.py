import os, math, random, statistics
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets

# --- import your code ---
from model_lightning import CBR_RNN, Transformer, LSTM
from word_tok import WordTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)

DATASET_PATH = "cbr_lightning/wikitext-103-tokenized"
TOKENIZER_PKL = "./tokenizer.pkl"
BATCH_SIZE = 2
SAMPLE_COUNT = 3
TOPK = 5

# -------- helpers --------
def build_causal_mask(seq_len, device):
    # Float mask with -inf above diagonal, 0 elsewhere (S,S)
    m = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    m = m.float()
    m = m.masked_fill(m == 1, float("-inf"))
    return m

def n_params(module):
    return sum(p.numel() for p in module.parameters())

def grad_stats(module):
    total, finite, zeros = 0, 0, 0
    norms = []
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.numel()
        finite += torch.isfinite(g).sum().item()
        zeros += (g == 0).sum().item()
        norms.append(g.norm(p=2).item())
    return dict(
        params_with_grad=(sum(1 for p in module.parameters() if p.grad is not None)),
        grad_elems=total,
        finite_frac=0.0 if total == 0 else finite/total,
        zero_frac=0.0 if total == 0 else zeros/total,
        grad_norm_mean=(float(statistics.fmean(norms)) if norms else 0.0),
        grad_norm_max=(float(max(norms)) if norms else 0.0)
    )

def logits_targets_loss(logits, targets):
    # logits: (S,B,V) or (B,S,V) ; targets: (S,B) or (B,S)
    if logits.dim() != 3:
        raise ValueError(f"Logits must be 3D, got {logits.shape}")
    if targets.dim() != 2:
        raise ValueError(f"Targets must be 2D, got {targets.shape}")

    if logits.shape[0] == targets.shape[0] and logits.shape[1] == targets.shape[1]:
        # (S,B,V) with (S,B)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        # per-position perplexity along time (average over batch)
        S, B, V = logits.shape
        with torch.no_grad():
            per_pos = []
            for t in range(S):
                l = F.cross_entropy(logits[t], targets[t])
                per_pos.append(torch.exp(l).item())
    else:
        # assume (B,S,V) with (B,S)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        B, S, V = logits.shape
        with torch.no_grad():
            per_pos = []
            for t in range(S):
                l = F.cross_entropy(logits[:, t, :], targets[:, t])
                per_pos.append(torch.exp(l).item())
    return loss, per_pos

def show_topk(logits_row, target_id, id2word, k=5):
    probs = torch.softmax(logits_row, dim=-1)
    top_p, top_i = torch.topk(probs, k)
    preds = [(id2word.get(i.item(), "<unk>"), float(p.item())) for p, i in zip(top_p, top_i)]
    tgt = id2word.get(int(target_id), "<unk>")
    return tgt, preds

# -------- dataset / tokenizer checks --------
print("Loading dataset:", DATASET_PATH)
ds = datasets.load_from_disk(DATASET_PATH)
print(ds)
print()

print("Loading tokenizer:", TOKENIZER_PKL)
import pickle
with open(TOKENIZER_PKL, "rb") as f:
    tok = pickle.load(f)
vocab_size = len(tok['word2idx'])
print(f"Tokenizer vocab_size: {vocab_size}")
print()

# small loader
loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
batch = next(iter(loader))
x = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["input_ids"]]).to(DEVICE)
y = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["target_ids"]]).to(DEVICE)
print('XXXXX')
print(x.shape)
# B, S = x.shape
S, B = x.shape
print(f"Batch shapes: x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
print(f"x min/max: {x.min().item()}/{x.max().item()}  y min/max: {y.min().item()}/{y.max().item()}")
print(f"Any negative ids? x:{(x<0).any().item()} y:{(y<0).any().item()}")
print(f"Max id < vocab_size? {int(max(x.max().item(), y.max().item()))+1} <= {vocab_size} -> {max(x.max().item(), y.max().item()) < vocab_size}")
print("Check next-token alignment on a few examples:")
for b in range(min(B,2)):
    ok = (x[b,1:] == y[b,:-1]).all().item()
    print(f"  sample {b}: y[t]==x[t+1]? {ok}")
print()

# -------- model configs --------
V = vocab_size
print("=== Building models ===")
cbr = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=S, compressed_dim=32).to(DEVICE).train()
trf = Transformer(vocab_size=V, d_model=256, n_heads=4, n_layers=2, d_ff=512,
                  max_seq_len=S, dropout=0.1, lr=1e-3, temperature=1.0, gumbel_softmax=False).to(DEVICE).train()
lstm = LSTM(vocab_size=V, embedding_dim=256, hidden_dim=512).to(DEVICE).train()

print(f"Params: CBR_RNN={n_params(cbr):,}  Transformer={n_params(trf):,}  LSTM={n_params(lstm):,}")
print()

# -------- forward/backward + detailed checks --------
def run_model_block(name, model, x, y):
    print(f"\n--- {name} ---")
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    if name == "CBR_RNN":
        # expects (S,B)
        # x = x.transpose(0,1)
        # Initialize cache properly - this is crucial for CBR_RNN
        cache = model.init_cache(x)
        logits, _ = model(x, initial_cache=cache)
        shape = tuple(logits.shape)  # (S,B,V)
        targets = y.transpose(0,1)  # (S,B)
        
    elif name == "Transformer":
        # expects (B,S) in your current code, returns (S,B,V)
        logits = model(x)
        shape = tuple(logits.shape)
        targets = y.transpose(0,1)  # (S,B)
    else:  # LSTM expects (B,S), returns (B,S,V)
        logits, _ = model(x)
        shape = tuple(logits.shape)
        targets = y  # (B,S)

    print("logits shape:", shape)
    # basic stats
    with torch.no_grad():
        finite = torch.isfinite(logits).all().item()
        mean = logits.mean().item()
        std = logits.std().item()
    print(f"logits finite? {finite}  mean={mean:.4f}  std={std:.4f}")

    # loss
    loss, per_pos_ppl = logits_targets_loss(logits, targets)
    print(f"loss={loss.item():.4f}  ppl={math.exp(loss.item()):.2f}")
    print("per-position ppl (first 5 / last 5):",
          [f"{p:.1f}" for p in per_pos_ppl[:5]], "...",
          [f"{p:.1f}" for p in per_pos_ppl[-5:]])

    # top-k qualitative samples
    id2word = tok['idx2word']
    with torch.no_grad():
        print("Top-k predictions on a few random positions:")
        if logits.shape == (S,B,logits.size(-1)):
            for _ in range(SAMPLE_COUNT):
                t = random.randrange(S)
                b = random.randrange(B)
                tgt, preds = show_topk(logits[t, b], targets[t, b].item(), id2word, k=TOPK)
                print(f"  (t={t}, b={b}) target='{tgt}'  preds={preds}")
        else:  # (B,S,V)
            for _ in range(SAMPLE_COUNT):
                t = random.randrange(S)
                b = random.randrange(B)
                tgt, preds = show_topk(logits[b, t], targets[b, t].item(), id2word, k=TOPK)
                print(f"  (t={t}, b={b}) target='{tgt}'  preds={preds}")

    # backward
    loss.backward()
    gs = grad_stats(model)
    print("grad stats:", gs)

    # detect vanishing/exploding patterns
    if gs["grad_norm_max"] > 1e3:
        print("⚠️ very large grad norm detected.")
    if gs["finite_frac"] < 1.0:
        print("❌ non-finite gradients present.")

    # weight stats (embeddings / head if present)
    with torch.no_grad():
        for name_, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                w = mod.weight
                print(f"embedding '{name_}' weight mean={w.mean().item():.4f} std={w.std().item():.4f}")
                break
        # output head
        head_w = None
        for name_, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == V:
                head_w = mod.weight
        if head_w is not None:
            print(f"output head weight mean={head_w.mean().item():.4f} std={head_w.std().item():.4f}")

    return float(loss.item())

# run once
loss_cbr = run_model_block("CBR_RNN", cbr, x, y)
loss_trf = run_model_block("Transformer", trf, x, y)
loss_lstm = run_model_block("LSTM", lstm, x, y)

# -------- micro overfit on a single tiny batch --------
def micro_overfit(name, model, x, y, steps=10, lr=1e-2):
    print(f"\n[{name}] micro-overfit {steps} steps on a single batch")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    
    # For CBR_RNN, we need to maintain a persistent cache across steps
    if name == "CBR_RNN":
        # Initialize cache once
        # x_input = x.transpose(0, 1)
        persistent_cache = model.init_cache(x)
        print(f"  CBR_RNN: initialized cache shapes: {[tuple(c.shape) for c in persistent_cache]}")
    
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        
        if name == "CBR_RNN":
            # Use the persistent cache but detach it to prevent gradient accumulation issues
            # while still allowing gradients to flow through the current step
            # cache_input = tuple(c.detach().requires_grad_(True) for c in persistent_cache)
            # logits, new_cache = model(x, initial_cache=cache_input)

            # loss, _ = logits_targets_loss(logits, y.transpose(0,1))
            cache = model.init_cache(x)
            logits, _ = model(x, initial_cache=cache)
            loss, _ = logits_targets_loss(logits, y.transpose(0,1))
            # Update persistent cache with new values (detached)
            # if new_cache is not None:
            #     persistent_cache = tuple(c.detach() for c in new_cache)
                
        elif name == "Transformer":
            logits = model(x)
            loss, _ = logits_targets_loss(logits, y.transpose(0,1))
        else:  # LSTM
            logits, _ = model(x)
            loss, _ = logits_targets_loss(logits, y)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 0.5)
        
        opt.step()
        losses.append(loss.item())
        print(f"  step {s+1}: loss {loss.item():.6f}")
        
        # Early stopping if loss becomes very small
        if loss.item() < 0.01:
            print(f"  Early stopping at step {s+1} (loss < 0.01)")
            break
    
    if len(losses) > 1:
        improvement = losses[0] - losses[-1]
        improvement_pct = improvement / losses[0] * 100
        if improvement > 0:
            print(f"✅ loss ↓ {losses[0]:.6f} → {losses[-1]:.6f} (improvement: {improvement:.6f}, {improvement_pct:.1f}%)")
        else:
            print(f"⚠️ did not overfit this batch (loss: {losses[0]:.6f} → {losses[-1]:.6f})")
    
    return losses

print("\n" + "="*60)
print("MICRO-OVERFITTING TESTS")
print("="*60)

# Test with different learning rates for CBR_RNN
cbr_losses_1 = micro_overfit("CBR_RNN", cbr, x, y, steps=10, lr=1e-2)
print(f"CBR_RNN lr=1e-2: {len([l for l in cbr_losses_1 if l < cbr_losses_1[0]])} improving steps")

# Reset model and try different lr
cbr2 = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=S, compressed_dim=32).to(DEVICE).train()
cbr_losses_2 = micro_overfit("CBR_RNN", cbr2, x, y, steps=10, lr=3e-3)
print(f"CBR_RNN lr=3e-3: {len([l for l in cbr_losses_2 if l < cbr_losses_2[0]])} improving steps")

# Compare with other models
trf_losses = micro_overfit("Transformer", trf, x, y, steps=10, lr=1e-2)
lstm_losses = micro_overfit("LSTM", lstm, x, y, steps=10, lr=1e-2)

print(f"\nComparison:")
print(f"CBR_RNN (lr=1e-2): {cbr_losses_1[0]:.6f} → {cbr_losses_1[-1]:.6f}")
print(f"CBR_RNN (lr=3e-3): {cbr_losses_2[0]:.6f} → {cbr_losses_2[-1]:.6f}")
print(f"Transformer:       {trf_losses[0]:.6f} → {trf_losses[-1]:.6f}")
print(f"LSTM:              {lstm_losses[0]:.6f} → {lstm_losses[-1]:.6f}")

print("\n✅ Deep diagnostic complete.")

# Additional CBR_RNN specific diagnostics
print("\n" + "="*60)
print("CBR_RNN SPECIFIC DIAGNOSTICS")
print("="*60)

# Check cache compression effectiveness
cbr_fresh = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=S, compressed_dim=32).to(DEVICE).train()
x_input = x.transpose(0, 1)
initial_cache = cbr_fresh.init_cache(x_input)
print(f"Initial cache shapes: {[tuple(c.shape) for c in initial_cache]}")

with torch.no_grad():
    logits, compressed_cache = cbr_fresh(x_input, initial_cache=initial_cache)
    if compressed_cache is not None:
        print(f"Compressed cache shapes: {[tuple(c.shape) for c in compressed_cache]}")
        print(f"Compression ratio: {S}/{cbr_fresh.compressed_dim} = {S/cbr_fresh.compressed_dim:.2f}x")
    else:
        print("No compressed cache returned")

# Check attention patterns
print(f"Multi-head attention config: {cbr_fresh.nheads} heads, {cbr_fresh.nhid} hidden dim")
print(f"Sequence length: {S}, Compressed dim: {cbr_fresh.compressed_dim}")

import sys
from contextlib import redirect_stdout

if __name__ == "__main__":
    out_path = "diagnostic_report.txt"
    with open(out_path, "w") as f:
        with redirect_stdout(f):
            # Rerun the micro-overfit tests for the report
            print("="*80)
            print("CBR_RNN OVERFITTING ANALYSIS REPORT")
            print("="*80)
            
            cbr_test = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=S, compressed_dim=1).to(DEVICE).train()
            
            print("Testing different learning rates:")
            for lr in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
                cbr_temp = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=S, compressed_dim=1).to(DEVICE).train()
                losses = micro_overfit("CBR_RNN", cbr_temp, x, y, steps=15, lr=lr)
                improvement = losses[0] - losses[-1] if len(losses) > 1 else 0
                print(f"lr={lr}: improvement = {improvement:.6f}")
            
            print("\nTesting different compressed_dim/sequence_length ratios:")
            seq_len = 128  # Fixed sequence length
            ratios_to_test = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]  # 1/32, 1/16, 1/8, 1/4, 1/2, 1/1
            
            for ratio in ratios_to_test:
                comp_dim = max(1, int(seq_len * ratio))
                print(f"\nTesting ratio {ratio:.4f} (compressed_dim={comp_dim}, sequence_length={seq_len})")
                cbr_temp = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=seq_len, compressed_dim=comp_dim).to(DEVICE).train()
                losses = micro_overfit("CBR_RNN", cbr_temp, x, y, steps=15, lr=1e-2)
                improvement = losses[0] - losses[-1] if len(losses) > 1 else 0
                improvement_pct = (improvement / losses[0] * 100) if losses[0] > 0 else 0
                final_loss = losses[-1] if len(losses) > 0 else float('inf')
                param_count = n_params(cbr_temp)
                print(f"  ratio={ratio:.4f} (comp_dim={comp_dim}): improvement={improvement:.6f} ({improvement_pct:.1f}%), final_loss={final_loss:.6f}, params={param_count:,}")
            
            print("\nCompression efficiency analysis:")
            print("Ratio\tComp_Dim\tMemory_Reduction\tParam_Overhead")
            seq_len = 128
            base_memory = seq_len * 512  # baseline memory for full sequence
            
            for ratio in [0.0625, 0.125, 0.25, 0.5, 1.0]:
                comp_dim = max(1, int(seq_len * ratio))
                compressed_memory = comp_dim * 512
                memory_reduction = (base_memory - compressed_memory) / base_memory * 100
                
                # Calculate additional parameters from compression layers
                cbr_temp = CBR_RNN(ntoken=V, ninp=256, nhid=512, nheads=4, seq_len=seq_len, compressed_dim=comp_dim).to(DEVICE)
                compress_params = (
                    cbr_temp.hidden_compress.weight.numel() + cbr_temp.hidden_compress.bias.numel() +
                    cbr_temp.key_compress.weight.numel() + cbr_temp.key_compress.bias.numel() +
                    cbr_temp.value_compress.weight.numel() + cbr_temp.value_compress.bias.numel()
                )
                
                print(f"{ratio:.4f}\t{comp_dim}\t\t{memory_reduction:.1f}%\t\t{compress_params:,}")

    print(f"Diagnostic report saved to {out_path}")