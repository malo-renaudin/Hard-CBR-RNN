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

from models import CBR_LM, Transformer_LM, LSTM_LM
from old import WordTokenizer
import torch
from clean_RTs import prepare_data_correct, run_paper_analysis_correct, fit_baseline_model, fit_full_model
import pandas as pd

stories = pd.read_csv('tests/test_datasets/all_stories.tok', sep = '\t')


def load_trained_model(model_type, checkpoint_path):

    model_type = model_type.lower()
    
    if model_type == "cbr":
        model_class = CBR_LM
    elif model_type == "transformer":
        model_class = Transformer_LM
    elif model_type == "lstm":
        model_class = LSTM_LM
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Expected one of ['cbr', 'transformer', 'lstm'].")

    model = model_class.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def compute_surprisal_window(lightning_model, stories_df, tokenizer, model_type,
                             chunk_size=64, step_size=1, num_samples=4):
    """
    Unified surprisal computation for CBR, Transformer, and LSTM models.
    
    Args:
        lightning_model: Trained PyTorch Lightning model (CBR_LM, Transformer_LM, or LSTM_LM)
        stories_df: DataFrame with columns ['word', 'zone', 'item']
        tokenizer: Tokenizer with .stoi mapping (word -> token ID)
        model_type: str, one of ['cbr', 'transformer', 'lstm']
        chunk_size: int, context window length
        step_size: int, step between overlapping windows
        num_samples: int, only used for CBR (kept for compatibility)
    
    Returns:
        DataFrame with ['word', 'zone', 'item', 'surprisal']
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model.to(device)
    lightning_model.eval()

    model = lightning_model.model
    results = []

    for story_id in sorted(stories_df['item'].unique()):
        story_data = stories_df[stories_df['item'] == story_id].sort_values('zone')
        words = story_data['word'].tolist()
        print(f"Processing story {story_id} ({len(words)} words)...")

        # Convert words to token IDs
        token_ids = [tokenizer.stoi.get(word, 0) for word in words]

        surprisal_sums = [0.0] * len(words)
        surprisal_counts = [0] * len(words)

        # Slide over words
        for start_idx in range(0, len(token_ids) - 1, step_size):
            end_idx = min(start_idx + chunk_size, len(token_ids))
            chunk_ids = token_ids[start_idx:end_idx]
            if len(chunk_ids) < 2:
                continue

            with torch.no_grad():
                # --- Transformer ---
                if model_type == "transformer":
                    input_tensor = torch.tensor(chunk_ids).unsqueeze(1).to(device)
                    forward_kwargs = {'src': input_tensor}
                    if hasattr(lightning_model, 'use_gumbel_softmax') and lightning_model.use_gumbel_softmax:
                        forward_kwargs.update({'temperature': 0.1, 'use_gumbel': False})
                    logits = model(**forward_kwargs)
                    log_probs = F.log_softmax(logits, dim=-1)

                    for i in range(1, len(chunk_ids)):
                        target_id = chunk_ids[i]
                        log_prob = log_probs[i-1, 0, target_id].item()
                        surprisal = -log_prob / math.log(2)
                        story_pos = start_idx + i
                        if story_pos < len(surprisal_sums):
                            surprisal_sums[story_pos] += surprisal
                            surprisal_counts[story_pos] += 1

                # --- CBR ---
                elif model_type == "cbr":
                    input_tensor = torch.tensor(chunk_ids).unsqueeze(1).to(device)
                    initial_cache = model.init_cache(input_tensor)
                    forward_kwargs = {'observation': input_tensor, 'initial_cache': initial_cache}
                    if hasattr(lightning_model, 'use_gumbel_softmax') and lightning_model.use_gumbel_softmax:
                        forward_kwargs.update({'temperature': 0.1, 'use_gumbel': False})
                    logits, _ = model(**forward_kwargs)
                    log_probs = F.log_softmax(logits, dim=-1)

                    for i in range(1, len(chunk_ids)):
                        target_id = chunk_ids[i]
                        log_prob = log_probs[i-1, 0, target_id].item()
                        surprisal = -log_prob / math.log(2)
                        story_pos = start_idx + i
                        if story_pos < len(surprisal_sums):
                            surprisal_sums[story_pos] += surprisal
                            surprisal_counts[story_pos] += 1

                # --- LSTM ---
                elif model_type == "lstm":
                    input_tensor = torch.tensor([chunk_ids], dtype=torch.long).to(device)
                    hidden = model.init_hidden(1, device)
                    logits, hidden = model(input_tensor, hidden)
                    log_probs = F.log_softmax(logits, dim=-1)

                    for i in range(1, len(chunk_ids)):
                        target_id = chunk_ids[i]
                        log_prob = log_probs[0, i-1, target_id].item()
                        surprisal = -log_prob / math.log(2)
                        story_pos = start_idx + i
                        if story_pos < len(surprisal_sums):
                            surprisal_sums[story_pos] += surprisal
                            surprisal_counts[story_pos] += 1

                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

        # Average surprisals
        story_surprisals = [
            (surprisal_sums[i] / surprisal_counts[i]) if surprisal_counts[i] > 0 else float('nan')
            for i in range(len(words))
        ]

        # Collect
        for idx, (_, row) in enumerate(story_data.iterrows()):
            results.append({
                'word': row['word'],
                'zone': row['zone'],
                'item': row['item'],
                'surprisal': story_surprisals[idx]
            })

    return pd.DataFrame(results)


tokenizer = WordTokenizer.load('tokenizer_ancien.json')


checkpoints = {
    'lstm_128': '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_lstm_000/lightning_logs/version_1851503/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_256' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_lstm_001/lightning_logs/version_1851549/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_512' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_lstm_002/lightning_logs/version_1851612/checkpoints/epoch=49-step=309500.ckpt',
    'lstm_1024' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_lstm_003/lightning_logs/version_1851441/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_128_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_000/lightning_logs/version_1356201/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_256_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_000/lightning_logs/version_1851737/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_512_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_001/lightning_logs/version_1356202/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_1024_1_false':'/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_001/lightning_logs/version_1851738/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_128_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_002/lightning_logs/version_1356203/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_256_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_002/lightning_logs/version_1851786/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_512_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_003/lightning_logs/version_1356204/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_1024_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_003/lightning_logs/version_1851787/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_004/lightning_logs/version_1356205/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_256_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_004/lightning_logs/version_1851788/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_512_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_005/lightning_logs/version_1356206/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_005/lightning_logs/version_1851789/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_006/lightning_logs/version_1356207/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_006/lightning_logs/version_1852086/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/final_models/job_transformer_007/lightning_logs/version_1356197/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_1024_8_true': '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_transformer_2_007/lightning_logs/version_1851705/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_000/lightning_logs/version_1364335/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_000/lightning_logs/version_1850824/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_001/lightning_logs/version_1364336/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_001/lightning_logs/version_1850825/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_002/lightning_logs/version_1364337/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_002/lightning_logs/version_1850826/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_003/lightning_logs/version_1364338/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_003/lightning_logs/version_1850827/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_004/lightning_logs/version_1364339/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_004/lightning_logs/version_1850828/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_005/lightning_logs/version_1364340/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_005/lightning_logs/version_1850865/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_006/lightning_logs/version_1364341/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_006/lightning_logs/version_1850866/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_007/lightning_logs/version_1364333/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_007/lightning_logs/version_1850817/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_0_5_exp' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_000/lightning_logs/version_1934821/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_0_1_linear' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_001/lightning_logs/version_1934822/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_0_5_linear': '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_002/lightning_logs/version_1934823/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_0_1_cosine':'/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_003/lightning_logs/version_1934824/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_0_5_cosine' : '/scratch2/mrenaudin/Hard-CBR-RNN/checkpoints/job_cbr_2_004/lightning_logs/version_1934763/checkpoints/epoch=49-step=309500.ckpt',

    # 'lstm_128' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_lstm_000/lightning_logs/version_1356004/checkpoints/epoch=49-step=309500.ckpt',
}

# Placeholder dictionary to store ΔLogLik
delta_loglik_dict = {}

# Load your reading time and surprisal data
# (You need to have these prepared for your analysis)
rt_data = pd.read_csv('tests/test_datasets/processed_RTs.tsv', sep='\t')

# Loop over checkpoints
for name, ckpt_path in checkpoints.items():
    print(f"\nRunning analysis for checkpoint: {name}")

    if any(x in name.lower() for x in ["transformer", "cbr", "lstm"]):
        model_type = next(x for x in ["transformer", "cbr", "lstm"] if x in name.lower())
    else:
        raise ValueError(f"❌ Could not infer model type from checkpoint name: {name}")
    
    model = load_trained_model(model_type, ckpt_path)
    
    surprisal_data = compute_surprisal_window(
        lightning_model=model,
        stories_df=stories,
        tokenizer=tokenizer,
        model_type=model_type  # "transformer", "cbr", or "lstm"
    )
    # Run ΔLogLik analysis
    results = run_paper_analysis_correct(rt_data, surprisal_data)
    
    # Store ΔLogLik
    delta_loglik_dict[name] = results['delta_loglik']

# Print results
print("\n=== ΔLogLik for all checkpoints ===")
for name, delta in delta_loglik_dict.items():
    print(f"{name}: {delta:.2f}")

with open("full_analysis.json", "w") as f:
    json.dump(delta_loglik_dict, f, indent=4)

print("\nΔLogLik results saved to delta_loglik_results.json")

# # Placeholder dictionary to store ΔLogLik
# delta_loglik_dict = {}

# # Load reading time data
# rt_data = pd.read_csv('tests/test_datasets/processed_RTs.tsv', sep='\t')

# # Prepare baseline data (without surprisal)
# print("Preparing baseline data...")
# data_for_baseline = prepare_data_correct(rt_data, surprisal_data=rt_data[['item', 'zone']].copy())
# # remove or ignore 'surprisal' column
# data_for_baseline = data_for_baseline.drop(columns=['surprisal'], errors='ignore')
# print('fitting baseline model')
# baseline_model = fit_baseline_model(data_for_baseline)

# # Loop over checkpoints
# for name, ckpt_path in checkpoints.items():
#     print(f"\nRunning analysis for checkpoint: {name}")

#     # Infer model type
#     if any(x in name.lower() for x in ["transformer", "cbr", "lstm"]):
#         model_type = next(x for x in ["transformer", "cbr", "lstm"] if x in name.lower())
#     else:
#         raise ValueError(f"❌ Could not infer model type from checkpoint name: {name}")
    
#     # Load trained model
#     model = load_trained_model(model_type, ckpt_path)
    
#     # Compute surprisal for this model
#     surprisal_data = compute_surprisal_window(
#         lightning_model=model,
#         stories_df=stories,
#         tokenizer=tokenizer,
#         model_type=model_type
#     )

#     # Merge surprisal into baseline data
#     data_with_surprisal = baseline_data.copy()
#     data_with_surprisal['current_surprisal'] = surprisal_data['surprisal'].values
#     data_with_surprisal['prev_surprisal'] = data_with_surprisal.groupby('item')['current_surprisal'].shift(1)
#     data_with_surprisal = data_with_surprisal.dropna(subset=['prev_surprisal']).reset_index(drop=True)

#     # Fit full model and compute ΔLogLik
#     results = fit_full_model(data_with_surprisal, baseline_model=baseline_model)

#     # Store ΔLogLik
#     delta_loglik_dict[name] = results['delta_loglik']

# # Print results
# print("\n=== ΔLogLik for all checkpoints ===")
# for name, delta in delta_loglik_dict.items():
#     print(f"{name}: {delta:.2f}")

# with open("test.json", "w") as f:
#     json.dump(delta_loglik_dict, f, indent=4)

# print("\nΔLogLik results saved to test.json")
