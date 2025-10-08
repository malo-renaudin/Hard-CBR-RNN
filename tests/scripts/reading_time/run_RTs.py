import pandas as pd
import numpy as np
import json
import torch.nn as nn
import torch
# if not hasattr(torch._dynamo, 'external_utils'):
#     import types
#     torch._dynamo.external_utils = types.ModuleType('external_utils')
#     torch._dynamo.external_utils.is_compiling = lambda: True

import pytorch_lightning as pl
import torch.nn.functional as F
import math

from lit_cbr import CBRLanguageModel
from grid_search import WordTokenizer
import torch
from clean_RTs import prepare_data_correct, compute_delta_loglik_correct, run_paper_analysis_correct
from grid_search import WordTokenizer  # assuming your tokenizer class is here
import pandas as pd

stories = pd.read_csv('/scratch2/mrenaudin/Hard-CBR-RNN/all_stories.tok', sep = '\t')

def load_trained_cbr(checkpoint_path):
    """Load the trained Lightning model"""
    model = CBRLanguageModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

from entire_transformer import SimpleTransformerLM
def load_trained_transformer(checkpoint_path):
    model = SimpleTransformerLM.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

from simple_lstm import SimpleLSTM_LM
def load_trained_lstm(checkpoint_path):
    model=SimpleLSTM_LM.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def compute_surprisal_window_transformer(lightning_model, stories_df, tokenizer, chunk_size=64, step_size=1):
    """
    Compute surprisal using overlapping chunks and averaging for SimpleTransformer model
    
    Args:
        lightning_model: Loaded Lightning SimpleTransformerLM
        stories_df: DataFrame with columns [word, zone, item]
        tokenizer: WordTokenizer object
        chunk_size: Sequence length used during training (64)
        step_size: Step size between overlapping windows
    
    Returns:
        DataFrame with added 'surprisal' column
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model.to(device)
    lightning_model.eval()  # Set to evaluation mode
    
    # Extract the actual SimpleTransformer model
    model = lightning_model.model
    
    results = []
    
    # Process each story
    for story_id in sorted(stories_df['item'].unique()):
        story_data = stories_df[stories_df['item'] == story_id].sort_values('zone')
        words = story_data['word'].tolist()
        
        print(f"Processing story {story_id} ({len(words)} words)...")
        
        # Convert words to token IDs using the tokenizer
        token_ids = []
        for word in words:
            token_id = tokenizer.stoi.get(word, 0)  # 0 = <unk>
            token_ids.append(token_id)
        
        # Initialize surprisal accumulation
        surprisal_sums = [0.0] * len(words)
        surprisal_counts = [0] * len(words)
        
        # Generate multiple overlapping windows
        # for start_offset in range(0, min(chunk_size, len(token_ids)), step_size):
        #     for start_idx in range(start_offset, len(token_ids) - 1, chunk_size):
        for start_idx in range(0, len(token_ids) - 1):
                end_idx = min(start_idx + chunk_size, len(token_ids))
                chunk_ids = token_ids[start_idx:end_idx]
                
                if len(chunk_ids) < 2:  # Need at least 2 tokens for prediction
                    continue
                
                # Convert to tensor [seq_len, batch_size=1]
                input_tensor = torch.tensor(chunk_ids).unsqueeze(1).to(device)
                
                # Forward pass
                with torch.no_grad():
                    forward_kwargs = {
                        'src': input_tensor
                    }
                    
                    # Add Gumbel parameters if model was trained with them
                    if hasattr(lightning_model, 'use_gumbel_softmax') and lightning_model.use_gumbel_softmax:
                        forward_kwargs.update({
                            'temperature': 0.1,
                            'use_gumbel': False  # Don't use Gumbel for surprisal computation
                        })
                    
                    # Get logits: [seq_len, batch_size=1, vocab_size]
                    logits = model(**forward_kwargs)
                    
                    # Compute log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Accumulate surprisals for this chunk
                    for i in range(1, len(chunk_ids)):  # Skip first token
                        target_token_id = chunk_ids[i]
                        
                        # Use prediction from step i-1 for token at step i
                        log_prob = log_probs[i-1, 0, target_token_id].item()
                        surprisal = -log_prob / math.log(2)  # Convert to bits
                        
                        # Map back to story position
                        story_pos = start_idx + i
                        if story_pos < len(surprisal_sums):
                            surprisal_sums[story_pos] += surprisal
                            surprisal_counts[story_pos] += 1
        
        # Compute averaged surprisals
        story_surprisals = []
        for i in range(len(words)):
            if surprisal_counts[i] > 0:
                avg_surprisal = surprisal_sums[i] / surprisal_counts[i]
                story_surprisals.append(avg_surprisal)
            else:
                story_surprisals.append(float('nan'))
        
        # Collect results for this story
        for idx, (_, row) in enumerate(story_data.iterrows()):
            results.append({
                'word': row['word'],
                'zone': row['zone'],
                'item': row['item'],
                'surprisal': story_surprisals[idx]
            })
    
    return pd.DataFrame(results)

def compute_surprisal_window_cbr(lightning_model, stories_df, tokenizer, chunk_size=64, step_size=1, num_samples=4):
    """
    Compute surprisal using overlapping chunks and averaging
    
    Args:
        lightning_model: Loaded Lightning CBRLanguageModel
        stories_df: DataFrame with columns [word, zone, item]
        tokenizer: WordTokenizer object
        chunk_size: Sequence length used during training (64)
        step_size: Step size between overlapping windows
        num_samples: Number of different starting positions to sample per token
    
    Returns:
        DataFrame with added 'surprisal' column
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model.to(device)
    
    # Extract the actual CBR_RNN model
    model = lightning_model.model
    
    results = []
    
    # Process each story
    for story_id in sorted(stories_df['item'].unique()):
        story_data = stories_df[stories_df['item'] == story_id].sort_values('zone')
        words = story_data['word'].tolist()
        
        print(f"Processing story {story_id} ({len(words)} words)...")
        
        # Convert words to token IDs
        token_ids = []
        for word in words:
            token_id = tokenizer.stoi.get(word, 0)  # 0 = <unk>
            token_ids.append(token_id)
        
        # Initialize surprisal accumulation
        surprisal_sums = [0.0] * len(words)
        surprisal_counts = [0] * len(words)
        
        # Generate multiple overlapping windows
        # for start_offset in range(0, min(chunk_size, len(token_ids)), step_size):
        #     for start_idx in range(start_offset, len(token_ids) - 1, chunk_size):
        for start_idx in range(0, len(token_ids) - 1):
                end_idx = min(start_idx + chunk_size, len(token_ids))
                chunk_ids = token_ids[start_idx:end_idx]
                
                if len(chunk_ids) < 2:  # Need at least 2 tokens
                    continue
                
                # Convert to tensor [chunk_len, batch_size=1]
                input_tensor = torch.tensor(chunk_ids).unsqueeze(1).to(device)
                
                # Initialize cache for this chunk
                initial_cache = model.init_cache(input_tensor)
                
                # Forward pass
                with torch.no_grad():
                    forward_kwargs = {
                        'observation': input_tensor,
                        'initial_cache': initial_cache
                    }
                    
                    if hasattr(lightning_model, 'use_gumbel_softmax') and lightning_model.use_gumbel_softmax:
                        forward_kwargs.update({
                            'temperature': 0.1,
                            'use_gumbel': False
                        })
                    
                    logits, states = model(**forward_kwargs)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # Accumulate surprisals for this chunk
                    for i in range(1, len(chunk_ids)):  # Skip first token
                        target_token_id = chunk_ids[i]
                        log_prob = log_probs[i-1, 0, target_token_id].item()
                        surprisal = -log_prob / math.log(2)
                        
                        # Map back to story position
                        story_pos = start_idx + i
                        if story_pos < len(surprisal_sums):
                            surprisal_sums[story_pos] += surprisal
                            surprisal_counts[story_pos] += 1
        
        # Compute averaged surprisals
        story_surprisals = []
        for i in range(len(words)):
            if surprisal_counts[i] > 0:
                avg_surprisal = surprisal_sums[i] / surprisal_counts[i]
                story_surprisals.append(avg_surprisal)
            else:
                story_surprisals.append(float('nan'))
        
        # Collect results for this story
        for idx, (_, row) in enumerate(story_data.iterrows()):
            results.append({
                'word': row['word'],
                'zone': row['zone'],
                'item': row['item'],
                'surprisal': story_surprisals[idx]
            })
    
    return pd.DataFrame(results)

def compute_surprisal_window_lstm(lightning_model, stories_df, tokenizer, chunk_size=64):
    """
    Compute surprisal using overlapping chunks for LSTM model
    
    Args:
        lightning_model: Loaded SimpleLSTM_LM (PyTorch Lightning module)
        stories_df: DataFrame with columns [word, zone, item]
        tokenizer: Tokenizer object with .stoi attribute (word -> token_id mapping)
        chunk_size: Sequence length used during training (default: 64)
    
    Returns:
        DataFrame with added 'surprisal' column
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model.to(device)
    lightning_model.eval()  # Set to evaluation mode
    
    # Extract the actual LSTM model
    model = lightning_model.model
    
    results = []
    
    # Process each story
    for story_id in sorted(stories_df['item'].unique()):
        story_data = stories_df[stories_df['item'] == story_id].sort_values('zone')
        words = story_data['word'].tolist()
        
        print(f"Processing story {story_id} ({len(words)} words)...")
        
        # Convert words to token IDs
        token_ids = []
        for word in words:
            token_id = tokenizer.stoi.get(word, 0)  # 0 = <unk>
            token_ids.append(token_id)
        
        # Initialize surprisal accumulation
        surprisal_sums = [0.0] * len(words)
        surprisal_counts = [0] * len(words)
        
        # Generate overlapping windows
        for start_idx in range(0, len(token_ids) - 1):
            end_idx = min(start_idx + chunk_size, len(token_ids))
            chunk_ids = token_ids[start_idx:end_idx]
            
            if len(chunk_ids) < 2:  # Need at least 2 tokens (input + target)
                continue
            
            # Convert to tensor [batch_size=1, seq_len]
            input_tensor = torch.tensor([chunk_ids], dtype=torch.long).to(device)
            
            # Initialize hidden state for this chunk
            batch_size = 1
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            with torch.no_grad():
                logits, hidden = model(input_tensor, hidden)
                # logits shape: [batch_size=1, seq_len, vocab_size]
                
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Accumulate surprisals for this chunk
                for i in range(1, len(chunk_ids)):  # Skip first token (no prediction yet)
                    target_token_id = chunk_ids[i]
                    # Get log prob for target token at position i-1 (predicting token i)
                    log_prob = log_probs[0, i-1, target_token_id].item()
                    surprisal = -log_prob / math.log(2)  # Convert to bits
                    
                    # Map back to story position
                    story_pos = start_idx + i
                    if story_pos < len(surprisal_sums):
                        surprisal_sums[story_pos] += surprisal
                        surprisal_counts[story_pos] += 1
        
        # Compute averaged surprisals
        story_surprisals = []
        for i in range(len(words)):
            if surprisal_counts[i] > 0:
                avg_surprisal = surprisal_sums[i] / surprisal_counts[i]
                story_surprisals.append(avg_surprisal)
            else:
                # First token or no valid windows
                story_surprisals.append(float('nan'))
        
        # Collect results for this story
        for idx, (_, row) in enumerate(story_data.iterrows()):
            results.append({
                'word': row['word'],
                'zone': row['zone'],
                'item': row['item'],
                'surprisal': story_surprisals[idx]
            })
    
    return pd.DataFrame(results)


tokenizer = WordTokenizer.load('tokenizer.json')
checkpoints = {
    'lstm_128': '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_000/lightning_logs/version_1851503/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_256' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_001/lightning_logs/version_1851549/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_512' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_002/lightning_logs/version_1851612/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_1024' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_003/lightning_logs/version_1851441/checkpoints/epoch=49-step=309500.ckpt'
    # 'transformer_128_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_000/lightning_logs/version_1356201/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_256_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_000/lightning_logs/version_1851737/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_512_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_001/lightning_logs/version_1356202/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_1024_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_001/lightning_logs/version_1851738/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_128_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_002/lightning_logs/version_1356203/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_256_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_002/lightning_logs/version_1851786/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_512_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_003/lightning_logs/version_1356204/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_1024_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_003/lightning_logs/version_1851787/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_004/lightning_logs/version_1356205/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_256_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_004/lightning_logs/version_1851788/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_512_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_005/lightning_logs/version_1356206/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_005/lightning_logs/version_1851789/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_006/lightning_logs/version_1356207/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_006/lightning_logs/version_1852086/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_007/lightning_logs/version_1356197/checkpoints/epoch=49-step=309500.ckpt',
    # 'transformer_1024_8_true': '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_007/lightning_logs/version_1851705/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_128_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_000/lightning_logs/version_1364335/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_256_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_000/lightning_logs/version_1850824/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_512_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_001/lightning_logs/version_1364336/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_1024_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_001/lightning_logs/version_1850825/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_128_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_002/lightning_logs/version_1364337/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_256_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_002/lightning_logs/version_1850826/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_512_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_003/lightning_logs/version_1364338/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_1024_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_003/lightning_logs/version_1850827/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_004/lightning_logs/version_1364339/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_256_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_004/lightning_logs/version_1850828/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_512_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_005/lightning_logs/version_1364340/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_005/lightning_logs/version_1850865/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_128_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_006/lightning_logs/version_1364341/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_006/lightning_logs/version_1850866/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_007/lightning_logs/version_1364333/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_1024_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_007/lightning_logs/version_1850817/checkpoints/epoch=49-step=309500.ckpt'
    # 'cbr_0_5_exp' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_000/lightning_logs/version_1934821/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_0_1_linear' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_001/lightning_logs/version_1934822/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_0_5_linear': '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_002/lightning_logs/version_1934823/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_0_1_cosine':'/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_003/lightning_logs/version_1934824/checkpoints/epoch=49-step=309500.ckpt',
    # 'cbr_0_5_cosine' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_004/lightning_logs/version_1934763/checkpoints/epoch=49-step=309500.ckpt',

    # 'lstm_128' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_lstm_000/lightning_logs/version_1356004/checkpoints/epoch=49-step=309500.ckpt',
}

# Placeholder dictionary to store ΔLogLik
delta_loglik_dict = {}

# Load your reading time and surprisal data
# (You need to have these prepared for your analysis)
rt_data = pd.read_csv('processed_RTs.tsv', sep='\t')

# Loop over checkpoints
for name, ckpt_path in checkpoints.items():
    print(f"\nRunning analysis for checkpoint: {name}")
    if name.startswith("transformer"):
        model = load_trained_transformer(ckpt_path)
        surprisal_data = compute_surprisal_window_transformer(model, stories, tokenizer=tokenizer)

    elif name.startswith("cbr"):
        model = load_trained_cbr(ckpt_path)
        surprisal_data = compute_surprisal_window_cbr(model, stories, tokenizer=tokenizer)
    
    elif name.startswith("lstm"):
        model = load_trained_lstm(ckpt_path)
        surprisal_data = compute_surprisal_window_lstm(model, stories, tokenizer=tokenizer)

    # Run ΔLogLik analysis
    results = run_paper_analysis_correct(rt_data, surprisal_data)
    
    # Store ΔLogLik
    delta_loglik_dict[name] = results['delta_loglik']

# Print results
print("\n=== ΔLogLik for all checkpoints ===")
for name, delta in delta_loglik_dict.items():
    print(f"{name}: {delta:.2f}")

with open("delta_loglik_results_lstm.json", "w") as f:
    json.dump(delta_loglik_dict, f, indent=4)

print("\nΔLogLik results saved to delta_loglik_results.json")