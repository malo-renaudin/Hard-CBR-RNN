import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
import copy

# Import your exact CBR_RNN and custom attention
from model_lightning import CBR_RNN
from model_lightning import MultiheadAttention as CustomMultiheadAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def create_cbr_with_pytorch_attention(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs):
    """Create CBR_RNN but replace custom attention with PyTorch standard"""
    model = CBR_RNN(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs)
    
    # Replace the custom attention with standard PyTorch attention
    model.multihead_attn = nn.MultiheadAttention(
        embed_dim=nhid, num_heads=nheads, batch_first=True, dropout=0.1
    )
    
    return model


def create_cbr_with_custom_attention_normal_softmax(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs):
    """Create CBR_RNN with custom attention but force normal softmax"""
    model = CBR_RNN(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs)
    # The attention is already custom, just ensure we use normal softmax
    model.gumbel_softmax = False
    model.temperature = 1.0
    return model


def create_cbr_no_attention(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs):
    """Create CBR_RNN but bypass attention entirely"""
    model = CBR_RNN(ntoken, ninp, nhid, nheads, seq_len, compressed_dim, **kwargs)
    
    # Monkey patch the forward method to skip attention
    original_forward = model.forward
    
    def forward_no_attention(self, observation, initial_cache=None, nheads=None, temperature=None, gumbel_softmax=None):
        seq_len = observation.size(0)
    
        if initial_cache is not None:
            hidden, key_cache, value_cache = initial_cache
        else:
            hidden, key_cache, value_cache = self.init_cache(observation)
        
        emb = self.drop(self.encoder(observation))
        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            
            # SKIP ATTENTION - just use query as attention output
            attn_output = query.squeeze(1)  # Remove the middle dimension
            query = query.squeeze(1)
            
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn_output, hidden)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i)
        
        decoded = self.decoder(hidden[-self.seq_len:]).transpose(0,1)
        cache = self.compress_cache(hidden, key_cache, value_cache)
        return decoded, cache
    
    # Bind the new method
    import types
    model.forward = types.MethodType(forward_no_attention, model)
    return model


def run_attention_test(name, model, x, y, steps=25, lr=1e-3):
    """Test a specific CBR_RNN attention configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    losses = []
    grad_norms = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Fresh cache every step - exactly as in your diagnostic
        cache = model.init_cache(x)
        
        if name == "PYTORCH_ATTENTION":
            # For PyTorch attention, don't pass temperature/gumbel_softmax
            def forward_pytorch_compatible(observation, initial_cache=None, **kwargs):
                seq_len = observation.size(0)
                if initial_cache is not None:
                    hidden, key_cache, value_cache = initial_cache
                else:
                    hidden, key_cache, value_cache = model.init_cache(observation)
                
                emb = model.drop(model.encoder(observation))
                for i in range(seq_len):
                    query = model.get_query(emb[i], hidden)
                    # Use standard PyTorch attention call (no temperature/gumbel_softmax)
                    attn_output, _ = model.multihead_attn(query, key_cache, value_cache, need_weights=False)
                    attn_output, query = attn_output.squeeze(1), query.squeeze(1)
                    key_cache_i, value_cache_i, hidden_i = model.intermediate_layers(i, emb, query, attn_output, hidden)
                    key_cache, value_cache, hidden = model.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i)
                
                decoded = model.decoder(hidden[-model.seq_len:]).transpose(0,1)
                cache = model.compress_cache(hidden, key_cache, value_cache)
                return decoded, cache
            
            # Temporarily replace forward method
            import types
            original_forward = model.forward
            model.forward = types.MethodType(forward_pytorch_compatible, model)
            logits, _ = model(x, initial_cache=cache)
            model.forward = original_forward  # Restore
        else:
            # Use normal forward method
            logits, _ = model(x, initial_cache=cache, 
                            nheads=model.nheads, 
                            temperature=model.temperature, 
                            gumbel_softmax=model.gumbel_softmax)
        
        # Compute loss - logits should be (B, S, V) format from your implementation
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = y.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        grad_norms.append(total_norm)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 5 == 0 or step == steps - 1:
            print(f"  Step {step+1:2d}: Loss = {loss.item():.6f}, Grad Norm = {total_norm:.4f}")
        
        # Early stopping
        if loss.item() < 0.1:
            print(f"  Early stopping at step {step+1} (loss < 0.1)")
            break
    
    # Analysis
    improvement = (losses[0] - losses[-1]) / losses[0] * 100 if len(losses) > 1 else 0
    converged = improvement > 10
    
    print(f"\nResults:")
    print(f"  Initial Loss: {losses[0]:.6f}")
    print(f"  Final Loss:   {losses[-1]:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Avg Grad Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    print(f"  Max Grad Norm: {max(grad_norms):.4f}")
    
    if improvement < 1:
        print(f"  ❌ POOR LEARNING")
    elif improvement < 10:
        print(f"  ⚠️  SLOW LEARNING")
    else:
        print(f"  ✅ GOOD LEARNING")
    
    return losses, improvement, converged


def test_attention_mechanisms(vocab_size, x, y):
    """Test different attention mechanisms with your exact CBR_RNN"""
    
    print("="*80)
    print("CBR_RNN ATTENTION MECHANISM DEBUGGING")
    print("="*80)
    
    base_config = {
        'ntoken': vocab_size,
        'ninp': 256,
        'nhid': 512,
        'nheads': 4,
        'seq_len': 128,
        'compressed_dim': 32,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'temperature': 1.0,
        'gumbel_softmax': False
    }
    
    test_configs = [
        {
            'name': 'NO_ATTENTION',
            'create_fn': create_cbr_no_attention,
            'config': base_config
        },
        {
            'name': 'PYTORCH_ATTENTION', 
            'create_fn': create_cbr_with_pytorch_attention,
            'config': base_config
        },
        {
            'name': 'CUSTOM_NORMAL_SOFTMAX',
            'create_fn': create_cbr_with_custom_attention_normal_softmax,
            'config': {**base_config, 'gumbel_softmax': False, 'temperature': 1.0}
        },
        {
            'name': 'CUSTOM_GUMBEL_TEMP_1.0',
            'create_fn': CBR_RNN,
            'config': {**base_config, 'gumbel_softmax': True, 'temperature': 1.0}
        },
        {
            'name': 'CUSTOM_GUMBEL_TEMP_0.5',
            'create_fn': CBR_RNN,
            'config': {**base_config, 'gumbel_softmax': True, 'temperature': 0.5}
        },
        {
            'name': 'CUSTOM_GUMBEL_TEMP_2.0',
            'create_fn': CBR_RNN,
            'config': {**base_config, 'gumbel_softmax': True, 'temperature': 2.0}
        },
        {
            'name': 'CUSTOM_GUMBEL_TEMP_5.0',
            'create_fn': CBR_RNN,
            'config': {**base_config, 'gumbel_softmax': True, 'temperature': 5.0}
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        try:
            print(f"\n{'-'*25} {test_config['name']} {'-'*25}")
            
            model = test_config['create_fn'](**test_config['config']).to(DEVICE)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,}")
            
            losses, improvement, converged = run_attention_test(
                test_config['name'], model, x, y, steps=25, lr=1e-3
            )
            
            results[test_config['name']] = {
                'losses': losses,
                'improvement': improvement,
                'converged': converged,
                'param_count': param_count
            }
            
        except Exception as e:
            print(f"ERROR in {test_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            results[test_config['name']] = {'error': str(e)}
    
    # Summary Report
    print("\n" + "="*80)
    print("ATTENTION DEBUGGING SUMMARY")
    print("="*80)
    
    working_configs = []
    failing_configs = []
    
    for name, result in results.items():
        if 'error' not in result:
            status = "✅ WORKS" if result['converged'] else "❌ FAILS"
            print(f"{name:25} | {status} | Improvement: {result['improvement']:5.1f}%")
            
            if result['converged']:
                working_configs.append(name)
            else:
                failing_configs.append(name)
        else:
            print(f"{name:25} | ❌ ERROR | {result['error']}")
            failing_configs.append(name)
    
    # Diagnosis
    print(f"\n📊 DIAGNOSIS:")
    
    if 'NO_ATTENTION' in working_configs:
        print("✅ Base CBR_RNN architecture works without attention")
        
        if 'PYTORCH_ATTENTION' in working_configs:
            print("✅ Standard PyTorch attention works with CBR_RNN")
            
            if 'CUSTOM_NORMAL_SOFTMAX' in working_configs:
                print("✅ Your custom attention works with normal softmax")
                print("🔍 Problem is likely with Gumbel softmax parameters")
                
                gumbel_working = [name for name in working_configs if 'GUMBEL' in name]
                if gumbel_working:
                    print(f"✅ Gumbel softmax works with these temperatures: {gumbel_working}")
                else:
                    print("❌ Gumbel softmax breaks learning entirely")
            else:
                print("❌ Your custom attention implementation has fundamental issues")
                print("🔍 Problem is in your scaled_dot_product_attention or MultiheadAttention code")
        else:
            print("❌ Even standard PyTorch attention fails - deeper CBR_RNN architecture issue")
    else:
        print("❌ Base CBR_RNN architecture has fundamental problems")
    
    return results


if __name__ == "__main__":
    # Load your data
    DATASET_PATH = "cbr_lightning/wikitext-103-tokenized" 
    TOKENIZER_PKL = "./tokenizer.pkl"
    BATCH_SIZE = 2
    
    # Load dataset and tokenizer
    import datasets
    import pickle
    ds = datasets.load_from_disk(DATASET_PATH)
    with open(TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)
    vocab_size = len(tok['word2idx'])
    
    # Get test batch
    loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(loader))
    x = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["input_ids"]]).to(DEVICE)
    y = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in batch["target_ids"]]).to(DEVICE)
    
    print(f"Data shape: x={x.shape}, y={y.shape}")
    print(f"Vocab size: {vocab_size}")
    
    # Run the attention debugging
    results = test_attention_mechanisms(vocab_size, x, y)