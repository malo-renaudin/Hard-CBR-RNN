import os, math, random, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
from collections import defaultdict
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)

class DiagnosticCBR_RNN(pl.LightningModule):
    """Flexible CBR_RNN for systematic debugging"""
    
    def __init__(self, ntoken, ninp, nhid, nheads, seq_len, compressed_dim, 
                 # Debug flags to control architecture
                 use_attention=True,
                 use_cache=True, 
                 use_compression=True,
                 use_intermediate_layers=True,
                 use_multihead=True,
                 max_seq_len=None,  # Limit sequence processing
                 simple_mode=False,  # Ultra-simple mode
                 dropout=0.1, learning_rate=1e-3):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Core parameters
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.nheads = nheads if use_multihead else 1
        self.seq_len = seq_len
        self.compressed_dim = compressed_dim
        self.max_seq_len = max_seq_len or seq_len
        
        # Debug flags
        self.use_attention = use_attention
        self.use_cache = use_cache
        self.use_compression = use_compression
        self.use_intermediate_layers = use_intermediate_layers
        self.simple_mode = simple_mode
        self.learning_rate = learning_rate
        
        # Basic layers
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        self.drop = nn.Dropout(dropout)
        
        if simple_mode:
            # Ultra-simple: just embedding -> linear -> output
            self.simple_transform = nn.Linear(ninp, nhid)
            self.init_weights_simple()
            return
        
        # Standard layers
        if use_intermediate_layers:
            self.q = nn.Linear(ninp + nhid, nhid)
            self.intermediate_h = nn.Linear(nhid * 3 + ninp, nhid * 4)
            self.final_h = nn.Linear(nhid * 4, nhid * 3)
            self.q_norm = nn.LayerNorm(nhid)
            self.int_norm = nn.LayerNorm(nhid * 4)
            self.f_norm = nn.LayerNorm(nhid * 3)
        else:
            # Simplified: direct embedding to hidden
            self.direct_transform = nn.Linear(ninp, nhid)
        
        if use_attention:
            if use_multihead and nheads > 1:
                # Use standard PyTorch MultiheadAttention to rule out custom implementation issues
                self.multihead_attn = nn.MultiheadAttention(
                    embed_dim=nhid, num_heads=self.nheads, batch_first=True, dropout=dropout
                )
            else:
                # Simple single-head attention
                self.attention_weights = nn.Linear(nhid, nhid)
        
        if use_compression and use_cache:
            # Compression layers
            self.hidden_pool = nn.AdaptiveAvgPool1d(compressed_dim)
            self.key_pool = nn.AdaptiveAvgPool1d(compressed_dim)
            self.value_pool = nn.AdaptiveAvgPool1d(compressed_dim)
            
            self.hidden_compress_norm = nn.LayerNorm(nhid)
            self.key_compress_norm = nn.LayerNorm(nhid)
            self.value_compress_norm = nn.LayerNorm(nhid)
        
        self.init_weights()
    
    def init_weights_simple(self):
        """Simple initialization for debugging"""
        nn.init.normal_(self.encoder.weight, mean=0, std=0.02)
        nn.init.normal_(self.decoder.weight, mean=0, std=0.02)
        nn.init.normal_(self.simple_transform.weight, mean=0, std=0.02)
        nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.simple_transform.bias)
    
    def init_weights(self):
        """Conservative weight initialization"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name or "decoder" in name:
                    nn.init.normal_(param, mean=0, std=0.02)  # Small init
                else:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Very small gain
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def init_cache(self, observation):
        """Initialize cache"""
        if not self.use_cache:
            return None
            
        if len(observation.size()) > 1:
            bsz = observation.size(-1)
        else:
            bsz = 1
        
        hidden = torch.zeros(self.compressed_dim, bsz, self.nhid, device=self.device)
        if self.use_attention:
            key_cache = torch.zeros(bsz, self.compressed_dim, self.nhid, device=self.device)
            value_cache = torch.zeros(bsz, self.compressed_dim, self.nhid, device=self.device)
            return hidden, key_cache, value_cache
        else:
            return (hidden,)
    
    def simple_forward(self, observation):
        """Ultra-simple forward pass for debugging"""
        # observation: (S, B)
        seq_len, batch_size = observation.shape
        
        # Process each timestep independently (no recurrence)
        outputs = []
        for i in range(min(seq_len, self.max_seq_len)):
            emb = self.encoder(observation[i])  # (B, ninp)
            hidden = self.simple_transform(emb)  # (B, nhid)
            output = self.decoder(hidden)  # (B, ntoken)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # (S, B, ntoken)
    
    def forward(self, observation, initial_cache=None):
        if self.simple_mode:
            return self.simple_forward(observation), None
        
        seq_len, batch_size = observation.shape
        actual_seq_len = min(seq_len, self.max_seq_len)
        
        if self.use_cache and initial_cache is not None:
            if self.use_attention:
                hidden, key_cache, value_cache = initial_cache
            else:
                hidden = initial_cache[0]
        else:
            if self.use_cache:
                cache_init = self.init_cache(observation)
                if self.use_attention:
                    hidden, key_cache, value_cache = cache_init
                else:
                    hidden = cache_init[0]
            else:
                hidden = torch.zeros(1, batch_size, self.nhid, device=self.device)
        
        emb = self.drop(self.encoder(observation))
        outputs = []
        
        for i in range(actual_seq_len):
            current_emb = emb[i]  # (B, ninp)
            
            if self.use_intermediate_layers:
                # Standard CBR_RNN processing
                if self.use_cache and len(hidden) > 0:
                    combined = torch.cat((current_emb, hidden[-1]), -1)
                else:
                    combined = torch.cat((current_emb, torch.zeros_like(current_emb).unsqueeze(0).expand(-1, -1, self.nhid).squeeze(0)), -1)
                
                query = self.drop(torch.tanh(self.q_norm(self.q(combined))))
                
                if self.use_attention:
                    query_expanded = query.unsqueeze(1)  # (B, 1, nhid)
                    
                    if hasattr(self, 'multihead_attn'):
                        # Use PyTorch's standard attention
                        if self.use_cache and key_cache.size(1) > 0:
                            attn_output, _ = self.multihead_attn(query_expanded, key_cache, value_cache)
                        else:
                            # No cache, self-attention on query
                            attn_output, _ = self.multihead_attn(query_expanded, query_expanded, query_expanded)
                        attn_output = attn_output.squeeze(1)  # (B, nhid)
                    else:
                        # Simple attention
                        attn_output = torch.tanh(self.attention_weights(query))
                else:
                    attn_output = query
                
                # Intermediate processing
                if self.use_cache and len(hidden) > 0:
                    intermediate_input = torch.cat((current_emb, query, attn_output, hidden[-1]), -1)
                else:
                    intermediate_input = torch.cat((current_emb, query, attn_output, torch.zeros(batch_size, self.nhid, device=self.device)), -1)
                
                intermediate = self.drop(torch.tanh(self.int_norm(self.intermediate_h(intermediate_input))))
                final_output = self.drop(torch.tanh(self.f_norm(self.final_h(intermediate))))
                key_cache_i, value_cache_i, hidden_i = final_output.split(self.nhid, dim=-1)
                
                # Update cache
                if self.use_cache:
                    hidden_i_expanded = hidden_i.unsqueeze(0)  # (1, B, nhid)
                    hidden = torch.cat((hidden, hidden_i_expanded), dim=0)
                    
                    if self.use_attention:
                        key_cache_i_expanded = key_cache_i.unsqueeze(1)  # (B, 1, nhid)
                        value_cache_i_expanded = value_cache_i.unsqueeze(1)  # (B, 1, nhid)
                        key_cache = torch.cat((key_cache, key_cache_i_expanded), dim=1)
                        value_cache = torch.cat((value_cache, value_cache_i_expanded), dim=1)
                
                current_hidden = hidden_i
            else:
                # Simplified processing
                current_hidden = self.direct_transform(current_emb)
            
            output = self.decoder(current_hidden)
            outputs.append(output)
        
        final_outputs = torch.stack(outputs, dim=0)  # (S, B, ntoken)
        
        # Prepare cache for return
        if self.use_cache:
            if self.use_compression:
                cache = self.compress_cache(hidden, key_cache, value_cache) if self.use_attention else (hidden,)
            else:
                cache = (hidden, key_cache, value_cache) if self.use_attention else (hidden,)
        else:
            cache = None
        
        return final_outputs, cache
    
    def compress_cache(self, hidden, key_cache, value_cache):
        """Compress cache using adaptive pooling"""
        if not self.use_compression:
            return hidden, key_cache, value_cache
        
        # Adaptive pooling compression
        hidden_reshaped = hidden.transpose(0, 1).transpose(1, 2)  # -> [batch, nhid, seq_len]
        hidden_pooled = self.hidden_pool(hidden_reshaped)  # -> [batch, nhid, compressed_dim]
        hidden_pooled = hidden_pooled.transpose(1, 2)  # -> [batch, compressed_dim, nhid]
        hidden_compressed = self.drop(torch.tanh(self.hidden_compress_norm(hidden_pooled)))
        hidden_compressed = hidden_compressed.transpose(0, 1)  # -> [compressed_dim, batch, nhid]
        
        key_transposed = key_cache.transpose(1, 2)  # -> [batch, nhid, seq_len]
        key_pooled = self.key_pool(key_transposed)  # -> [batch, nhid, compressed_dim]
        key_compressed = key_pooled.transpose(1, 2)  # -> [batch, compressed_dim, nhid]
        key_compressed = self.drop(torch.tanh(self.key_compress_norm(key_compressed)))
        
        value_transposed = value_cache.transpose(1, 2)  # -> [batch, nhid, seq_len]
        value_pooled = self.value_pool(value_transposed)  # -> [batch, nhid, compressed_dim]
        value_compressed = value_pooled.transpose(1, 2)  # -> [batch, compressed_dim, nhid]
        value_compressed = self.drop(torch.tanh(self.value_compress_norm(value_compressed)))
        
        return hidden_compressed, key_compressed, value_compressed
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)


def run_overfit_test(model_name, model, x, y, steps=20, lr=1e-3, clip_norm=1.0):
    """Test model's ability to overfit a single batch"""
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"Learning rate: {lr}, Gradient clip: {clip_norm}")
    print(f"{'='*60}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    losses = []
    grad_norms = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Fresh cache every step to avoid persistence issues
        if hasattr(model, 'init_cache'):
            cache = model.init_cache(x)
            logits, _ = model(x, initial_cache=cache)
        else:
            logits, _ = model(x)
        
        # Ensure correct format for loss computation
        if logits.dim() == 3:
            if logits.shape[0] == x.shape[0]:  # (S, B, V)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
            else:  # (B, S, V)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.transpose(0, 1).contiguous().view(-1)
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        loss.backward()
        
        # Analyze gradients
        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        grad_norms.append(total_norm)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 5 == 0 or step == steps - 1:
            print(f"  Step {step+1:2d}: Loss = {loss.item():.6f}, Grad Norm = {total_norm:.4f}")
        
        # Early stopping if loss becomes very small
        if loss.item() < 0.1:
            print(f"  Early stopping at step {step+1} (loss < 0.1)")
            break
    
    # Analysis
    if len(losses) > 1:
        improvement = losses[0] - losses[-1]
        improvement_pct = improvement / losses[0] * 100 if losses[0] > 0 else 0
        
        print(f"\nRESULTS:")
        print(f"  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss:   {losses[-1]:.6f}")
        print(f"  Improvement:  {improvement:.6f} ({improvement_pct:.1f}%)")
        print(f"  Converged:    {'YES' if improvement_pct > 10 else 'NO'}")
        print(f"  Avg grad norm: {sum(grad_norms)/len(grad_norms):.4f}")
        print(f"  Max grad norm: {max(grad_norms):.4f}")
        
        if improvement_pct < 1:
            print(f"  ❌ POOR LEARNING - improvement < 1%")
        elif improvement_pct < 10:
            print(f"  ⚠️  SLOW LEARNING - improvement < 10%")
        else:
            print(f"  ✅ GOOD LEARNING - improvement > 10%")
    
    return losses, grad_norms


def systematic_debug(vocab_size, x, y):
    """Run systematic debugging tests"""
    print("="*80)
    print("CBR_RNN SYSTEMATIC DEBUGGING")
    print("="*80)
    
    base_config = {
        'ntoken': vocab_size,
        'ninp': 256,
        'nhid': 512,
        'nheads': 4,
        'seq_len': 128,
        'compressed_dim': 32,
        'dropout': 0.1
    }
    
    test_cases = [
        # Test 1: Ultra-simple baseline
        {
            'name': 'SIMPLE_MODE',
            'config': {**base_config, 'simple_mode': True},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 2: No attention
        {
            'name': 'NO_ATTENTION',
            'config': {**base_config, 'use_attention': False, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 3: No cache
        {
            'name': 'NO_CACHE',
            'config': {**base_config, 'use_cache': False, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 4: Short sequences
        {
            'name': 'SHORT_SEQUENCES',
            'config': {**base_config, 'max_seq_len': 10, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 5: Single head attention
        {
            'name': 'SINGLE_HEAD',
            'config': {**base_config, 'use_multihead': False, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 6: No intermediate layers
        {
            'name': 'NO_INTERMEDIATE',
            'config': {**base_config, 'use_intermediate_layers': False, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 7: No compression
        {
            'name': 'NO_COMPRESSION',
            'config': {**base_config, 'use_compression': False, 'simple_mode': False},
            'lr': 1e-3,
            'clip': 1.0
        },
        
        # Test 8: Lower learning rate
        {
            'name': 'LOW_LR',
            'config': base_config,
            'lr': 1e-4,
            'clip': 1.0
        },
        
        # Test 9: Higher gradient clipping
        {
            'name': 'HIGH_CLIP',
            'config': base_config,
            'lr': 1e-3,
            'clip': 0.1
        },
        
        # Test 10: Everything simplified
        {
            'name': 'MINIMAL_CBR',
            'config': {
                **base_config,
                'use_attention': True,
                'use_cache': True,
                'use_compression': False,
                'use_intermediate_layers': False,
                'use_multihead': False,
                'max_seq_len': 20,
                'nhid': 128,
                'compressed_dim': 8
            },
            'lr': 1e-3,
            'clip': 0.5
        }
    ]
    
    results = {}
    
    for test in test_cases:
        try:
            print(f"\n{'='*20} {test['name']} {'='*20}")
            model = DiagnosticCBR_RNN(**test['config']).to(DEVICE)
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,}")
            
            losses, grad_norms = run_overfit_test(
                test['name'], model, x, y, 
                steps=25, lr=test['lr'], clip_norm=test['clip']
            )
            
            results[test['name']] = {
                'losses': losses,
                'grad_norms': grad_norms,
                'converged': (losses[0] - losses[-1]) / losses[0] > 0.1 if len(losses) > 1 else False,
                'param_count': param_count
            }
            
        except Exception as e:
            print(f"ERROR in {test['name']}: {e}")
            results[test['name']] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    working_configs = []
    for name, result in results.items():
        if 'error' not in result:
            converged = result['converged']
            final_loss = result['losses'][-1] if result['losses'] else float('inf')
            improvement = ((result['losses'][0] - result['losses'][-1]) / result['losses'][0] * 100) if len(result['losses']) > 1 else 0
            
            status = "✅ WORKS" if converged else "❌ FAILS"
            print(f"{name:15} | {status} | Final Loss: {final_loss:.4f} | Improvement: {improvement:.1f}% | Params: {result['param_count']:,}")
            
            if converged:
                working_configs.append(name)
        else:
            print(f"{name:15} | ❌ ERROR | {result['error']}")
    
    print(f"\nWorking configurations: {working_configs}")
    
    return results


if __name__ == "__main__":
    # Load your data here - adjust paths as needed
    DATASET_PATH = "cbr_lightning/wikitext-103-tokenized"
    TOKENIZER_PKL = "./tokenizer.pkl"
    BATCH_SIZE = 2
    
    # Load dataset and tokenizer
    ds = datasets.load_from_disk(DATASET_PATH)
    import pickle
    with open(TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)
    vocab_size = len(tok['word2idx'])
    
    # Get a small batch for testing
    loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(loader))
    x = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["input_ids"]]).to(DEVICE)
    y = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["target_ids"]]).to(DEVICE)
    
    print(f"Data shape: x={x.shape}, y={y.shape}")
    print(f"Vocab size: {vocab_size}")
    
    # Run systematic debugging
    results = systematic_debug(vocab_size, x, y)