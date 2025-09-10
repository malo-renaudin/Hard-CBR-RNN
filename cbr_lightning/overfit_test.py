import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import datasets
import math

# Import all models from model_lightning
from model_lightning import CBR_RNN, Transformer, LSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_overfitting_test(name, model, x, y, steps=30, lr=1e-3):
    """Run overfitting test on a single batch"""
    print(f"\n{'='*70}")
    print(f"OVERFITTING TEST: {name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*70}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    losses = []
    grad_norms = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass based on model type
        if "CBR_RNN" in name:
            # CBR_RNN expects (S, B) and uses cache
            cache = model.init_cache(x)
            
            if "GUMBEL" in name:
                logits, _ = model(x, initial_cache=cache, 
                                nheads=model.nheads, 
                                temperature=model.temperature, 
                                gumbel_softmax=True)
            else:
                logits, _ = model(x, initial_cache=cache,
                                nheads=model.nheads,
                                temperature=1.0,
                                gumbel_softmax=False)
            
            # logits are (B, S, V) from your implementation
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y.reshape(-1)
            
        elif "Transformer" in name:
            # Transformer expects (B, S) 
            x_input = x.transpose(0, 1)  # (S, B) -> (B, S)
            
            if "GUMBEL" in name:
                logits = model(x_input, temperature=model.temperature, gumbel_softmax=True)
            else:
                logits = model(x_input, temperature=1.0, gumbel_softmax=False)
            
            # Transformer returns (S, B, V), transpose to (B, S, V)
            logits = logits.transpose(0, 1)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y.reshape(-1)
            
        elif "LSTM" in name:
            # LSTM expects (B, S)
            x_input = x.transpose(0, 1)  # (S, B) -> (B, S)
            logits, _ = model(x_input)
            
            # LSTM returns (B, S, V)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 5 == 0 or step == steps - 1:
            print(f"  Step {step+1:2d}: Loss = {loss.item():.6f}, Grad Norm = {total_norm:.4f}")
        
        # Early stopping if converged
        if step > 5 and loss.item() < 0.5:
            print(f"  Early stopping at step {step+1} (loss < 0.5)")
            break
    
    # Analysis
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    converged = improvement > 30  # Higher threshold for good overfitting
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm = max(grad_norms)
    
    print(f"\n📊 RESULTS:")
    print(f"  Initial Loss:    {initial_loss:.6f}")
    print(f"  Final Loss:      {final_loss:.6f}")  
    print(f"  Improvement:     {improvement:.1f}%")
    print(f"  Avg Grad Norm:   {avg_grad_norm:.4f}")
    print(f"  Max Grad Norm:   {max_grad_norm:.4f}")
    
    if improvement > 50:
        print(f"  🏆 EXCELLENT OVERFITTING")
    elif improvement > 30:
        print(f"  ✅ GOOD OVERFITTING")
    elif improvement > 10:
        print(f"  ⚠️  MODERATE OVERFITTING")  
    else:
        print(f"  ❌ POOR OVERFITTING")
    
    return {
        'losses': losses,
        'improvement': improvement,
        'converged': converged,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'avg_grad_norm': avg_grad_norm,
        'max_grad_norm': max_grad_norm,
        'param_count': count_parameters(model)
    }


def create_matched_models(vocab_size, target_params=45_000_000):
    """Create models with approximately equal parameter counts"""
    
    models = {}
    
    # CBR_RNN with normal softmax
    cbr_normal = CBR_RNN(
        ntoken=vocab_size,
        ninp=256,
        nhid=512, 
        nheads=4,
        seq_len=128,
        compressed_dim=32,
        dropout=0.1,
        temperature=1.0,
        gumbel_softmax=False
    ).to(DEVICE)
    
    models['CBR_RNN_NORMAL'] = cbr_normal
    cbr_params = count_parameters(cbr_normal)
    print(f"CBR_RNN parameters: {cbr_params:,}")
    
    # CBR_RNN with Gumbel softmax  
    cbr_gumbel = CBR_RNN(
        ntoken=vocab_size,
        ninp=256,
        nhid=512,
        nheads=4, 
        seq_len=128,
        compressed_dim=32,
        dropout=0.1,
        temperature=1.0,
        gumbel_softmax=True
    ).to(DEVICE)
    
    models['CBR_RNN_GUMBEL'] = cbr_gumbel
    
    # Now match other models to CBR_RNN parameter count
    target_params = cbr_params
    
    # Transformer - adjust d_model and d_ff to match parameters
    # Rough formula: params ≈ vocab_size * d_model + d_model^2 * (4 + 4*n_layers)
    # Solve for d_model given target_params
    d_model = 384  # Start with this and adjust
    d_ff = d_model * 4
    
    transformer_normal = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=6, 
        d_ff=d_ff,
        max_seq_len=128,
        dropout=0.1,
        temperature=1.0,
        lr=1e-2,
        gumbel_softmax=False
    ).to(DEVICE)
    
    models['Transformer_NORMAL'] = transformer_normal
    transformer_params = count_parameters(transformer_normal)
    print(f"Transformer parameters: {transformer_params:,}")
    
    # Transformer with Gumbel
    transformer_gumbel = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        n_layers=6,
        d_ff=d_ff,
        max_seq_len=128,
        dropout=0.1,
        temperature=1.0,
        lr=1e-2,
        gumbel_softmax=True
    ).to(DEVICE)
    
    models['Transformer_GUMBEL'] = transformer_gumbel
    
    # LSTM - adjust hidden_dim to match parameters  
    # Rough formula: params ≈ vocab_size * embed_dim + 4 * hidden_dim * (embed_dim + hidden_dim + 1)
    hidden_dim = 512  # Start with this
    embedding_dim = 256
    
    lstm_model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=1e-2
    ).to(DEVICE)
    
    models['LSTM'] = lstm_model
    lstm_params = count_parameters(lstm_model)
    print(f"LSTM parameters: {lstm_params:,}")
    
    return models


def comprehensive_model_comparison(vocab_size, x, y):
    """Compare all models with matched parameter counts"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON - EQUAL PARAMETERS")
    print("="*80)
    
    # Create matched models
    models = create_matched_models(vocab_size)
    
    # Test each model
    results = {}
    
    test_order = [
        'CBR_RNN_NORMAL',
        'CBR_RNN_GUMBEL', 
        'Transformer_NORMAL',
        'Transformer_GUMBEL',
        'LSTM'
    ]
    
    for model_name in test_order:
        try:
            model = models[model_name]
            result = run_overfitting_test(model_name, model, x, y, steps=30, lr=1e-2)
            results[model_name] = result
            
        except Exception as e:
            print(f"\n❌ ERROR in {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")  
    print("="*80)
    
    print(f"{'Model':<20} | {'Params':<10} | {'Initial':<8} | {'Final':<8} | {'Improve':<7} | {'Status'}")
    print("-" * 80)
    
    for model_name in test_order:
        result = results[model_name]
        if 'error' not in result:
            params = f"{result['param_count']:,}"[:10]
            initial = f"{result['initial_loss']:.4f}"
            final = f"{result['final_loss']:.4f}"
            improve = f"{result['improvement']:.1f}%"
            
            if result['improvement'] > 50:
                status = "🏆 EXCELLENT"
            elif result['improvement'] > 30:
                status = "✅ GOOD"
            elif result['improvement'] > 10:
                status = "⚠️  MODERATE"
            else:
                status = "❌ POOR"
            
            print(f"{model_name:<20} | {params:<10} | {initial:<8} | {final:<8} | {improve:<7} | {status}")
        else:
            print(f"{model_name:<20} | {'ERROR':<10} | {'--':<8} | {'--':<8} | {'--':<7} | ❌ FAILED")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    
    # Find best performing model
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['improvement'])
        best_improvement = valid_results[best_model]['improvement']
        print(f"🥇 Best performing: {best_model} ({best_improvement:.1f}% improvement)")
        
        # Compare CBR_RNN variants
        cbr_normal = valid_results.get('CBR_RNN_NORMAL', {})
        cbr_gumbel = valid_results.get('CBR_RNN_GUMBEL', {})
        
        if cbr_normal and cbr_gumbel:
            normal_perf = cbr_normal['improvement']
            gumbel_perf = cbr_gumbel['improvement'] 
            print(f"🎯 CBR_RNN Normal vs Gumbel: {normal_perf:.1f}% vs {gumbel_perf:.1f}%")
            
            if abs(normal_perf - gumbel_perf) < 5:
                print("   → Gumbel softmax has minimal impact on CBR_RNN")
            elif gumbel_perf > normal_perf:
                print("   → Gumbel softmax helps CBR_RNN performance") 
            else:
                print("   → Normal softmax works better for CBR_RNN")
        
        # Compare Transformer variants
        trans_normal = valid_results.get('Transformer_NORMAL', {})
        trans_gumbel = valid_results.get('Transformer_GUMBEL', {})
        
        if trans_normal and trans_gumbel:
            normal_perf = trans_normal['improvement']
            gumbel_perf = trans_gumbel['improvement']
            print(f"🎯 Transformer Normal vs Gumbel: {normal_perf:.1f}% vs {gumbel_perf:.1f}%")
    
    return results


if __name__ == "__main__":
    # Load data
    DATASET_PATH = "cbr_lightning/wikitext-103-tokenized"
    TOKENIZER_PKL = "./tokenizer.pkl"
    BATCH_SIZE = 2
    
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
    
    # Run comprehensive comparison
    results = comprehensive_model_comparison(vocab_size, x, y)
    
    # Save results to file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_results_{timestamp}.txt"
    test_order = [
        'CBR_RNN_NORMAL',
        'CBR_RNN_GUMBEL', 
        'Transformer_NORMAL',
        'Transformer_GUMBEL',
        'LSTM'
    ]
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Data shape: x={x.shape}, y={y.shape}\n")
        f.write(f"Vocab size: {vocab_size}\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} | {'Params':<10} | {'Initial':<8} | {'Final':<8} | {'Improve':<7} | {'Status'}\n")
        f.write("-" * 80 + "\n")
        
        for model_name in test_order:
            result = results[model_name]
            if 'error' not in result:
                params = f"{result['param_count']:,}"[:10]
                initial = f"{result['initial_loss']:.4f}"
                final = f"{result['final_loss']:.4f}"
                improve = f"{result['improvement']:.1f}%"
                
                if result['improvement'] > 50:
                    status = "EXCELLENT"
                elif result['improvement'] > 30:
                    status = "GOOD"
                elif result['improvement'] > 10:
                    status = "MODERATE"
                else:
                    status = "POOR"
                
                f.write(f"{model_name:<20} | {params:<10} | {initial:<8} | {final:<8} | {improve:<7} | {status}\n")
            else:
                f.write(f"{model_name:<20} | {'ERROR':<10} | {'--':<8} | {'--':<8} | {'--':<7} | FAILED\n")
        
        # Detailed results
        f.write("\n\nDETAILED RESULTS:\n")
        f.write("="*50 + "\n")
        
        for model_name in test_order:
            result = results[model_name]
            f.write(f"\n{model_name}:\n")
            f.write("-" * 30 + "\n")
            
            if 'error' not in result:
                f.write(f"  Parameters: {result['param_count']:,}\n")
                f.write(f"  Initial Loss: {result['initial_loss']:.6f}\n")
                f.write(f"  Final Loss: {result['final_loss']:.6f}\n")
                f.write(f"  Improvement: {result['improvement']:.1f}%\n")
                f.write(f"  Avg Grad Norm: {result['avg_grad_norm']:.4f}\n")
                f.write(f"  Max Grad Norm: {result['max_grad_norm']:.4f}\n")
                f.write(f"  Converged: {result['converged']}\n")
                f.write(f"  Loss trajectory: {[f'{l:.4f}' for l in result['losses'][:10]]}...\n")
            else:
                f.write(f"  Error: {result['error']}\n")
    
    print(f"\nResults saved to: {filename}")
    
