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
from torch.utils.data import Dataset, DataLoader
import string
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from original_cbr import CBR_RNN
from simple_lstm import SimpleLSTM
from entire_transformer import SimpleTransformer

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


def load_tokenizer(tokenizer_path):
    """Load tokenizer from JSON file"""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    return tokenizer_data['stoi'], tokenizer_data['itos']

class BLiMPDataset(Dataset):
    def __init__(self, blimp_subset, stoi):
        self.dataset = load_dataset("nyu-mll/blimp", blimp_subset, split = 'train')
        # self.dictionary = dictionary
        self.encoded_pairs = []

        for example in self.dataset:
            sentence_good = example['sentence_good']
            sentence_bad = example['sentence_bad']
            sentence_good = sentence_good.rstrip(string.punctuation)
            sentence_bad = sentence_bad.rstrip(string.punctuation)
            encoded_good = [int(stoi.get(word.lower(), 0))  for word in sentence_good.split()]
            #encoded_good = [self.dictionary.word2idx.get(word) for word in sentence_good.split()]
            encoded_bad = [int(stoi.get(word.lower(), 0))  for word in sentence_bad.split()]
            #encoded_bad = [self.dictionary.word2idx.get(word) for word in sentence_bad.split()]
            self.encoded_pairs.append({
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "encoded_good": torch.tensor(encoded_good, dtype=torch.long),
                "encoded_bad": torch.tensor(encoded_bad, dtype=torch.long),
            })

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        return self.encoded_pairs[idx]
    
def collate_fn(batch):
    encoded_good_sequences = [item['encoded_good'] for item in batch]
    encoded_bad_sequences = [item['encoded_bad'] for item in batch]
    sentence_good = [item['sentence_good'] for item in batch]
    sentence_bad = [item['sentence_bad'] for item in batch]
    return {
        'sentence_bad':sentence_bad,
        'sentence_good': sentence_good,
        'encoded_good': pad_sequence(encoded_good_sequences, batch_first=True),
        'encoded_bad': pad_sequence(encoded_bad_sequences, batch_first=True)
    }

def compute_nll(model, sentence_tensor, device='cuda'):
    """
    Compute the negative log-likelihood of a sentence.
    Works with SimpleLSTM_LM, SimpleTransformerLM, and CBRLanguageModel.
    
    Args:
        model: SimpleLSTM_LM, SimpleTransformerLM, or CBRLanguageModel
        sentence_tensor: torch.LongTensor of shape (seq_len,) containing token indices
        device: device to run computation on
        
    Returns:
        nll: negative log-likelihood (scalar tensor)
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension: (seq_len,) -> (1, seq_len)
        sentence = sentence_tensor.unsqueeze(0).to(device)
        
        # Prepare input and target
        # Input: all tokens except the last one
        # Target: all tokens except the first one
        input_seq = sentence[:, :-1]  # (1, seq_len-1)
        target_seq = sentence[:, 1:]   # (1, seq_len-1)
        
        # Check model type and forward accordingly
        if isinstance(model.model, SimpleLSTM):
            # LSTM model
            batch_size = 1
            hidden = model.model.init_hidden(batch_size, device)
            output, _ = model.model(input_seq, hidden)  # (1, seq_len-1, vocab_size)
            
        elif isinstance(model.model, SimpleTransformer):
            # Transformer model - needs [seq_len, batch_size] format
            input_seq_transposed = input_seq.transpose(0, 1)  # (seq_len-1, 1)
            output = model.model(input_seq_transposed)  # (seq_len-1, 1, vocab_size)
            output = output.transpose(0, 1)  # (1, seq_len-1, vocab_size)
            
        elif isinstance(model.model, CBR_RNN):
            # CBR model - needs [seq_len, batch_size] format and cache initialization
            input_seq_transposed = input_seq.transpose(0, 1)  # (seq_len-1, 1)
            initial_cache = model.model.init_cache(input_seq_transposed)
            output, _ = model.model(input_seq_transposed, initial_cache)  # (seq_len-1, 1, vocab_size)
            output = output.transpose(0, 1)  # (1, seq_len-1, vocab_size)
            
        else:
            raise ValueError(f"Unknown model type: {type(model.model)}")
        
        # Compute NLL
        output_flat = output.reshape(-1, output.size(-1))  # (seq_len-1, vocab_size)
        target_flat = target_seq.reshape(-1)  # (seq_len-1,)
        
        # CrossEntropyLoss with reduction='sum' gives total NLL
        criterion = nn.CrossEntropyLoss(reduction='sum')
        nll = criterion(output_flat, target_flat)
        
    return nll


def evaluate_blimp_pair(model, encoded_good, encoded_bad, device='cuda'):
    """
    Evaluate a BLiMP sentence pair.
    Works with LSTM, Transformer, and CBR models.
    
    Returns:
        correct: True if model prefers grammatical sentence (lower NLL)
        nll_good: NLL of grammatical sentence
        nll_bad: NLL of ungrammatical sentence
    """
    nll_good = compute_nll(model, encoded_good, device)
    nll_bad = compute_nll(model, encoded_bad, device)
    
    # Model is correct if grammatical sentence has lower NLL
    correct = nll_good < nll_bad
    
    return correct, nll_good.item(), nll_bad.item()


def evaluate_blimp_dataset(model, blimp_dataset, device='cuda', verbose=True):
    """
    Evaluate model on entire BLiMP dataset.
    Works with LSTM, Transformer, and CBR models.
    """
    correct_count = 0
    total_count = len(blimp_dataset)
    
    for i in range(total_count):
        item = blimp_dataset[i]
        correct, nll_good, nll_bad = evaluate_blimp_pair(
            model, 
            item['encoded_good'], 
            item['encoded_bad'], 
            device
        )
        
        if correct:
            correct_count += 1
            
        if verbose and (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{total_count} ({correct_count}/{i+1} correct)")
    
    accuracy = correct_count / total_count
    if verbose:
        print(f"\nBLiMP Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    return accuracy

blimp_tasks = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
tokenizer_path = 'tokenizer.json'


checkpoints = {
    # 'lstm_128': '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_000/lightning_logs/version_1851503/checkpoints/epoch=49-step=309500.ckpt',
    # 'lstm_256' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_001/lightning_logs/version_1851549/checkpoints/epoch=49-step=309500.ckpt',
    # 'lstm_512' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_002/lightning_logs/version_1851612/checkpoints/epoch=49-step=309500.ckpt',
    # 'lstm_1024' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_lstm_003/lightning_logs/version_1851441/checkpoints/epoch=49-step=309500.ckpt',
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
    'transformer_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_005/lightning_logs/version_1851789/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_006/lightning_logs/version_1356207/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_006/lightning_logs/version_1852086/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/final_models/job_transformer_007/lightning_logs/version_1356197/checkpoints/epoch=49-step=309500.ckpt',
    'transformer_1024_8_true': '/scratch2/mrenaudin/Hard-CBR-RNN/job_transformer_2_007/lightning_logs/version_1851705/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_000/lightning_logs/version_1364335/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_000/lightning_logs/version_1850824/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_001/lightning_logs/version_1364336/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_1_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_001/lightning_logs/version_1850825/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_002/lightning_logs/version_1364337/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_002/lightning_logs/version_1850826/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_003/lightning_logs/version_1364338/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_8_false' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_003/lightning_logs/version_1850827/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_004/lightning_logs/version_1364339/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_004/lightning_logs/version_1850828/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_005/lightning_logs/version_1364340/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_1_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_005/lightning_logs/version_1850865/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_128_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_006/lightning_logs/version_1364341/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_256_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_006/lightning_logs/version_1850866/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_512_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_007/lightning_logs/version_1364333/checkpoints/epoch=49-step=309500.ckpt',
    'cbr_1024_8_true' : '/scratch2/mrenaudin/Hard-CBR-RNN/job_cbr_2_007/lightning_logs/version_1850817/checkpoints/epoch=49-step=309500.ckpt'
   
}

import json
from tqdm import tqdm
import os

def evaluate_all_models_on_blimp(checkpoints, blimp_tasks, tokenizer_path, device='cuda', save_path='blimp_results.json'):
    """
    Evaluate all models on all BLiMP tasks and save results.
    
    Args:
        checkpoints: Dictionary mapping model names to checkpoint paths
        blimp_tasks: List of BLiMP task names
        tokenizer_path: Path to tokenizer JSON file
        device: Device to run evaluation on
        save_path: Path to save results JSON
    
    Returns:
        results: Dictionary with structure {model_name: {task_name: accuracy}}
    """
    # Load tokenizer
    print("Loading tokenizer...")
    stoi, itos = load_tokenizer(tokenizer_path)
    
    # Initialize results dictionary
    results = {}
    
    # Iterate through all models
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*80}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found at {checkpoint_path}, skipping...")
            results[model_name] = {task: None for task in blimp_tasks}
            continue
        
        # Load the appropriate model based on model name
        try:
            if model_name.startswith('lstm'):
                model = load_trained_lstm(checkpoint_path)
            elif model_name.startswith('transformer'):
                model = load_trained_transformer(checkpoint_path)
            elif model_name.startswith('cbr'):
                model = load_trained_cbr(checkpoint_path)
            else:
                print(f"WARNING: Unknown model type for {model_name}, skipping...")
                results[model_name] = {task: None for task in blimp_tasks}
                continue
            
            model = model.to(device)
            print(f"Model loaded successfully!")
            
        except Exception as e:
            print(f"ERROR loading model {model_name}: {e}")
            results[model_name] = {task: None for task in blimp_tasks}
            continue
        
        # Initialize results for this model
        results[model_name] = {}
        
        # Evaluate on each BLiMP task
        for task_name in tqdm(blimp_tasks, desc=f"BLiMP tasks for {model_name}"):
            try:
                # Load BLiMP dataset for this task
                blimp_dataset = BLiMPDataset(task_name, stoi)
                
                # Evaluate
                accuracy = evaluate_blimp_dataset(
                    model=model,
                    blimp_dataset=blimp_dataset,
                    device=device,
                    verbose=False
                )
                
                results[model_name][task_name] = accuracy
                print(f"  {task_name}: {accuracy:.2%}")
                
            except Exception as e:
                print(f"  ERROR on task {task_name}: {e}")
                results[model_name][task_name] = None
        
        # Compute average accuracy for this model (excluding None values)
        valid_accuracies = [acc for acc in results[model_name].values() if acc is not None]
        if valid_accuracies:
            avg_accuracy = sum(valid_accuracies) / len(valid_accuracies)
            results[model_name]['average'] = avg_accuracy
            print(f"\n  Average accuracy for {model_name}: {avg_accuracy:.2%}")
        else:
            results[model_name]['average'] = None
            print(f"\n  No valid results for {model_name}")
        
        # Save results after each model (in case of crashes)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {save_path}")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")
    
    return results


def print_summary_table(results):
    """Print a summary table of results"""
    import pandas as pd
    
    # Create DataFrame for easier viewing
    df_data = []
    for model_name, task_results in results.items():
        row = {'model': model_name}
        row.update(task_results)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Print overall averages
    print("\nOverall Model Performance:")
    print("-" * 50)
    for model_name in results.keys():
        avg = results[model_name].get('average', None)
        if avg is not None:
            print(f"{model_name:40s}: {avg:.2%}")
        else:
            print(f"{model_name:40s}: No valid results")
    
    return df


# Run the evaluation
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = evaluate_all_models_on_blimp(
        checkpoints=checkpoints,
        blimp_tasks=blimp_tasks,
        tokenizer_path=tokenizer_path,
        device=device,
        save_path='blimp_results_all_models_2.json'
    )
    
    # Print summary
    df = print_summary_table(results)
    
    # Save DataFrame as CSV for easy analysis
    df.to_csv('blimp_results_summary_2.csv', index=False)
    print("\nResults also saved to: blimp_results_summary.csv")