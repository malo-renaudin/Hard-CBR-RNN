import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
import json
import random
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from grid_search.data_utils import WordTokenizer
from models.lstm import SimpleLSTM

sns.set_style("whitegrid")

class CBR_RNN(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, nheads=1, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)       
        self.q = nn.Linear(ninp+nhid, nhid)
        self.intermediate_h = nn.Linear(nhid*4, nhid*4)
        self.final_h = nn.Linear(nhid*4, nhid*3)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = nn.LayerNorm(nhid)
        self.int_norm = nn.LayerNorm(nhid * 4)
        self.f_norm = nn.LayerNorm(nhid * 3)            
        
        self.nhid = nhid
        self.nheads = nheads
        self.head_dim = nhid // nheads if nheads > 1 else nhid
        self.attn_div_factor = np.sqrt(self.head_dim)
        
        assert nhid % nheads == 0 if nheads > 1 else True, "nhid must be divisible by nheads"
        self.init_weights()

    def init_weights(self):
        # Embedding layer - smaller std for stability
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        
        # Output layer - specialized for large vocabulary
        # Use scaled normal initialization for large vocab (50K)
        output_std = (2.0 / (self.nhid + self.encoder.num_embeddings)) ** 0.5
        nn.init.normal_(self.decoder.weight, mean=0.0, std=output_std)
        nn.init.zeros_(self.decoder.bias)
        
        # Query layer - Kaiming initialization for ReLU activation
        nn.init.kaiming_uniform_(self.q.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.q.bias)
        
        # Intermediate layer - He initialization with residual scaling
        nn.init.kaiming_uniform_(self.intermediate_h.weight, mode='fan_in', nonlinearity='relu')
        # Scale down for residual-like behavior
        self.intermediate_h.weight.data *= 0.5
        nn.init.zeros_(self.intermediate_h.bias)
        
        # Final layer - Xavier for linear transformation before split
        nn.init.xavier_uniform_(self.final_h.weight, gain=1.0)
        nn.init.zeros_(self.final_h.bias)
        
        # Layer norm parameters (if they have learnable params)
        for layer_norm in [self.q_norm, self.int_norm, self.f_norm]:
            if hasattr(layer_norm, 'weight'):
                nn.init.ones_(layer_norm.weight)
            if hasattr(layer_norm, 'bias'):
                nn.init.zeros_(layer_norm.bias)
        


    def forward_with_attention_fixed(self, observation, initial_cache, temperature=0.1, use_gumbel=True, hard=False):
        hidden_init, key_cache_init, value_cache_init = initial_cache
        # observation = observation.squeeze(2)
        print('observation shape', observation.shape)
        seq_len, batch_size= observation.shape
        device = observation.device
        
        # Ensure we're in eval mode for deterministic behavior
        self.eval()
        
        # Pre-allocate all state tensors with explicit dtype
        states = torch.zeros(seq_len + 1, batch_size, self.nhid, 
                            device=device, dtype=torch.float32)
        keys = torch.zeros(seq_len + 1, batch_size, self.nhid, 
                        device=device, dtype=torch.float32)  
        values = torch.zeros(seq_len + 1, batch_size, self.nhid, 
                            device=device, dtype=torch.float32)
        
        # Initialize with explicit copying
        states[0].copy_(hidden_init.squeeze(0))
        keys[0].copy_(key_cache_init.squeeze(0))
        values[0].copy_(value_cache_init.squeeze(0))
        
        # Get embeddings WITHOUT dropout in eval mode
        emb = self.encoder(observation)
        
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), 
                                        device=device, dtype=torch.float32), diagonal=1)
        
        # Store attention weights
        attention_history = []
        attention_scores_history = []
        for i in range(seq_len):
            cache_len = i + 1
            curr_emb, curr_hidden = emb[i], states[i]
            
            # Query computation - NO dropout in eval mode
            concat_input = torch.cat([curr_emb, curr_hidden], dim=-1)
            query_raw = self.q(concat_input)
            query_norm = self.q_norm(query_raw)
            query = F.relu(query_norm)
            
            # Unified multi-head attention with explicit contiguous operations
            current_keys = keys[:cache_len].transpose(0, 1).contiguous()
            current_values = values[:cache_len].transpose(0, 1).contiguous()
            
            # Reshape for multi-head
            q_mh = query.view(batch_size, self.nheads, self.head_dim)
            k_mh = current_keys.view(batch_size, cache_len, self.nheads, self.head_dim)
            v_mh = current_values.view(batch_size, cache_len, self.nheads, self.head_dim)
            
            # Compute attention scores
            attn_scores = torch.einsum('bnh,bcnh->bnc', q_mh, k_mh)
            
            # Apply mask and scaling
            masked_scores = (attn_scores + causal_mask[i, :cache_len].unsqueeze(0).unsqueeze(0)) / self.attn_div_factor
            attention_scores_history.append(masked_scores.detach().cpu().clone())
            # FIXED: Proper deterministic attention
            if use_gumbel:
                # # True hard attention (deterministic argmax)
                # attn_indices = torch.argmax(masked_scores, dim=-1)
                # attn_weights = F.one_hot(attn_indices, num_classes=cache_len).float()
                attn_weights= F.gumbel_softmax(masked_scores, tau=temperature, hard=hard, dim=-1)
            else:
                # Standard softmax (deterministic)
                attn_weights = F.softmax(masked_scores/temperature, dim=-1)
            
            # CAPTURE ATTENTION WEIGHTS - ensure they're detached and cloned
            attention_history.append(attn_weights.detach().cpu().clone())
            
            # Compute attention output
            attn = torch.einsum('bnc,bcnh->bnh', attn_weights, v_mh).contiguous().view(batch_size, self.nhid)
            
            # Process through network layers - NO dropout in eval mode
            intermediate_input = torch.cat([curr_emb, query, attn, curr_hidden], dim=-1)
            intermediate_raw = self.intermediate_h(intermediate_input)
            intermediate = F.relu(self.int_norm(intermediate_raw))
            
            final_raw = self.final_h(intermediate)
            final_output = F.relu(self.f_norm(final_raw))
            key_i, value_i, hidden_i = final_output.split(self.nhid, dim=-1)
            
            # Update states
            states[i + 1].copy_(hidden_i)
            keys[i + 1].copy_(key_i)
            values[i + 1].copy_(value_i)
        
        
        
        return self.decoder(states[1:]), states, attention_history, attention_scores_history

    def init_cache(self, observation):
        device = observation.device
        bsz = observation.size(-1) if len(observation.size()) > 1 else 1
        return tuple(torch.zeros(1, bsz, self.nhid, device=device) for _ in range(3))
    
    def set_parameters(self, val):
        for module in [self.q, self.intermediate_h, self.final_h, self.encoder, self.decoder]:
            for param in module.parameters():
                param.data.fill_(val)

    def randomize_parameters(self):
        for module in [self.q, self.intermediate_h, self.final_h]:
            for param in module.parameters():
                nn.init.uniform_(param, -0.1, 0.1)




class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with attention tracking capability"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, temperature=1.0, 
                                    use_gumbel=False, hard=False):
        """
        Q, K, V: [batch_size, n_heads, seq_len, d_k]
        Returns: output, attention_weights
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        if use_gumbel:
            attention_weights = F.gumbel_softmax(scores / temperature, 
                                                tau=1.0, hard=hard, dim=-1)
        else:
            attention_weights = F.softmax(scores / temperature, dim=-1)
        
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None, temperature=1.0, 
               use_gumbel=False, hard=False):
        """
        query, key, value: [batch_size, seq_len, d_model]
        Returns: output, attention_weights
        """
        batch_size, seq_len, d_model = query.shape
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q=Q, K=K, V=V, mask=mask, temperature=temperature, 
            use_gumbel=use_gumbel, hard=hard
        )
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights


class TransformerLayer(nn.Module):
    """Transformer layer with attention tracking"""
    
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, temperature=1.0, use_gumbel=False, hard=False):
        attn_output, attn_weights = self.self_attn(
            x, x, x, mask=mask, temperature=temperature, 
            use_gumbel=use_gumbel, hard=hard
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class SimpleTransformer(nn.Module):
    """Transformer with attention tracking for evaluation"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(64, d_model)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, src, temperature=1.0, use_gumbel=False, hard=False):
        """Standard forward pass (for training)"""
        src = src.transpose(0, 1)
        batch_size, seq_len = src.shape
        device = src.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        causal_mask = self.create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        for layer in self.layers:
            x, _ = layer(x=x, mask=causal_mask, temperature=temperature, 
                        use_gumbel=use_gumbel, hard=hard)
        
        logits = self.output_projection(x).transpose(0, 1)
        
        return logits
    
    def forward_with_attention_tracking(self, src, temperature=1.0, 
                                       use_gumbel=False, hard=False, layer_idx=-1):
        """
        Forward pass with attention tracking (for evaluation/analysis).
        
        Args:
            src: Input tensor [seq_len, batch_size]
            temperature: Temperature for softmax/gumbel
            use_gumbel: Whether to use Gumbel-Softmax
            hard: Whether to use hard Gumbel-Softmax
            layer_idx: Which layer to track attention from (-1 = last layer, 0 = first layer)
        
        Returns:
            logits: Output logits [seq_len, batch_size, vocab_size]
            attention_history: List of attention weights for each position
                              Each element: [batch_size, n_heads, 1, attended_positions]
        """
        self.eval()
        
        src = src.transpose(0, 1)  # [batch_size, seq_len]
        batch_size, seq_len = src.shape
        device = src.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings (no dropout in eval mode)
        token_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        
        # Create full causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Store attention for the specified layer
        attention_history = []
        
        # Pass through all layers
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x=x, mask=causal_mask, temperature=temperature,
                                   use_gumbel=use_gumbel, hard=hard)
            
            # Store attention from the specified layer
            if i == layer_idx or (layer_idx == -1 and i == self.num_layers - 1):
                # attn_weights shape: [batch_size, n_heads, seq_len, seq_len]
                # We want to extract attention for each query position separately
                for t in range(seq_len):
                    # Extract attention for query position t
                    # Shape: [batch_size, n_heads, attended_positions (0:t+1)]
                    attn_at_t = attn_weights[:, :, t, :t+1]
                    attention_history.append(attn_at_t.detach().cpu().clone())
        
        # Project to vocabulary
        logits = self.output_projection(x).transpose(0, 1)
        
        return logits, attention_history



def load_tokenizer(data_dir, tokenizer_path):
    """Load tokenizer from JSON file"""
    tokenizer = WordTokenizer(data_dir)
    tokenizer.load(tokenizer_path)
    return tokenizer





class NounPPDataset(Dataset):
    def __init__(self, nounpp_file, tokenizer):
        self.sentences = []
        self.conditions = []
        self.correct = []
        self.wrong = []
        self.encoded_sentences = []
        self.encoded_correct = []
        self.encoded_wrong = []

        with open(nounpp_file, "r") as f:
            for line in f:
                line = line.split()
                sentence = line[1:7]
                condition = " ".join(line[7:9])
                wrong = line[9]
                correct = line[6]
                
                # Encode each word individually
                encoded_sentence = [
                    tokenizer.stoi.get(word.lower(), 0)  # Get token ID directly
                    for word in sentence
                ]
                encoded_correct = tokenizer.stoi.get(correct.lower(), 0)
                encoded_wrong = tokenizer.stoi.get(wrong.lower(), 0)
                
                self.sentences.append(sentence)
                self.conditions.append(condition)
                self.correct.append(correct)
                self.wrong.append(wrong)
                self.encoded_sentences.append(encoded_sentence)  # Now a list of ints
                self.encoded_correct.append(encoded_correct)      # Now an int
                self.encoded_wrong.append(encoded_wrong)          # Now an int

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "encoded_sentence": torch.tensor(
                self.encoded_sentences[idx], dtype=torch.long
            ),  # [6] 
            "correct": self.correct[idx],
            "encoded_correct": self.encoded_correct[idx],  # scalar int
            "wrong": self.wrong[idx],
            "encoded_wrong": self.encoded_wrong[idx],      # scalar int
            "condition": self.conditions[idx],
        }
        


def collate_fn_nounpp(batch):
    sentences = [item["sentence"] for item in batch]
    encoded_sentences = torch.stack([item["encoded_sentence"] for item in batch])
    
    # Convert lists of scalars to tensors
    encoded_correct = torch.tensor([item["encoded_correct"] for item in batch], dtype=torch.long)
    encoded_wrong = torch.tensor([item["encoded_wrong"] for item in batch], dtype=torch.long)
    
    correct = [item["correct"] for item in batch]
    wrong = [item["wrong"] for item in batch]
    conditions = [item["condition"] for item in batch]

    return {
        "sentence": sentences,
        "encoded_sentence": encoded_sentences,  # [batch_size, 6]
        "correct": correct,
        "encoded_correct": encoded_correct,     # [batch_size]
        "wrong": wrong,
        "encoded_wrong": encoded_wrong,         # [batch_size]
        "condition": conditions,
    }

def eval_with_attention_analysis(model, test_dataloader, temperature, use_gumbel, track_attention=True):
    """
    Efficient evaluation: Process first 5 words to predict verb at position 5.
    Works with both CBR-RNN and Transformer models.
    """
    condition_accuracies = defaultdict(int)
    condition_counts = defaultdict(int)
    sentence_details = []
    attention_by_condition = defaultdict(list) if track_attention else None
    
    model.eval()
    
    # Detect model type
    is_transformer = hasattr(model, 'token_embedding')
    
    with torch.no_grad():
        for batch in test_dataloader:
            written = batch["sentence"]
            sentence = batch["encoded_sentence"]  # [batch_size, 6]
            correct = batch["encoded_correct"]
            wrong = batch["encoded_wrong"]
            condition = batch["condition"]
            batch_size = sentence.size(0)
            limit = sentence.size(1)-1
            # Process only first 5 words (positions 0-4) to predict verb at position 5
            context = sentence[:, :limit].transpose(0, 1)  # [5, batch_size]
            if track_attention:
                if is_transformer:
                    # Transformer: use forward_with_attention_tracking
                    out, attn_history = model.forward_with_attention_tracking(
                        context, temperature=temperature, 
                        use_gumbel=use_gumbel, hard=False
                    )
                else:
                    # CBR-RNN: use forward_with_attention_fixed
                    cache = model.init_cache(context)
                    out, _, attn_history, _ = model.forward_with_attention_fixed(
                        context, cache, temperature=temperature, 
                        use_gumbel=use_gumbel, hard=False
                    )
            else:
                if is_transformer:
                    out = model(context, temperature=temperature, use_gumbel=use_gumbel)
                else:
                    cache = model.init_cache(context)
                    out, _ = model(context, cache, temperature=temperature, use_gumbel=use_gumbel)
            # Use output at position 4 (last position) which predicts position 5 (verb)
            log_probs = torch.nn.functional.log_softmax(out[-1], dim=-1)
            correct_log_probs = log_probs[torch.arange(batch_size), correct]
            wrong_log_probs = log_probs[torch.arange(batch_size), wrong]
            correct_predictions = correct_log_probs >= wrong_log_probs

            for i in range(batch_size):
                cond = condition[i]
                pred = correct_predictions[i].item()
                
                condition_counts[cond] += 1
                condition_accuracies[cond] += pred

                sentence_details.append({
                    "sentence": written[i],
                    "condition": cond,
                    "correct_log_prob": correct_log_probs[i].item(),
                    "wrong_log_prob": wrong_log_probs[i].item(),
                    "model_prefers_correct": pred,
                })
                
                if track_attention:
                    seq_len = context.shape[0]  # 5
                    attention_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
                    
                    for t, weights in enumerate(attn_history):
                        cache_len = t + 1
                        # weights shape: [batch_size, n_heads, cache_len]
                        # Extract for sample i, head 0 (or average across heads)
                        attention_matrix[t, :cache_len] = weights[i, 0, :cache_len].cpu().numpy()
                    
                    attention_by_condition[cond].append(attention_matrix)

    final_accuracies = {
        cond: condition_accuracies[cond] / condition_counts[cond]
        for cond in condition_accuracies
    }
    
    if track_attention:
        return final_accuracies, attention_by_condition, sentence_details
    else:
        return final_accuracies, None, sentence_details




def plot_accuracy_comparison(accuracies, save_path=None):
    """Simple bar plot of accuracy across conditions."""
    conditions = sorted(accuracies.keys())
    acc_values = [accuracies[c] for c in conditions]
    
    # Color by correctness
    colors = []
    for c in conditions:
        parts = c.split()
        if (parts[0] == 'singular' and parts[1] == 'singular') or \
           (parts[0] == 'plural' and parts[1] == 'plural'):
            colors.append('#2ecc71')
        else:
            colors.append('#e74c3c')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(conditions)), acc_values, color=colors, 
                  alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, acc in zip(bars, acc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, 
               alpha=0.7, label='Chance')
    
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Condition (Noun → Verb)', fontsize=13, fontweight='bold')
    ax.set_title('Model Accuracy on Subject-Verb Agreement', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in conditions], fontsize=10)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()



def plot_attention_grid_with_verb(attention_stats, dataset, save_path=None):
    """Plot 2x2 grid of attention heatmaps."""
    conditions = [
        ('singular singular', 'Singular → Singular'),
        ('singular plural', 'Singular → Plural'),
        ('plural singular', 'Plural → Singular'),
        ('plural plural', 'Plural → Plural')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (cond_key, cond_label) in enumerate(conditions):
        ax = axes[idx]
        
        if cond_key not in attention_stats or attention_stats[cond_key][0] is None:
            ax.set_title(f'{cond_label}\n(No data)', fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        mean_att, _, attn_matrices = attention_stats[cond_key]
        n_samples = len(attn_matrices)
        
        # Get example words for labels
        words = None
        if hasattr(dataset, 'sentences') and hasattr(dataset, 'conditions'):
            for i, c in enumerate(dataset.conditions):
                if c == cond_key:
                    words = dataset.sentences[i][:5]  # Only first 5 words
                    break
        
        # Plot heatmap with blue gradient
        sns.heatmap(mean_att, ax=ax, cmap='Blues', vmin=0, vmax=1, 
                   cbar_kws={'label': 'Attention'}, square=True,
                   xticklabels=words if words else range(mean_att.shape[1]),
                   yticklabels=words if words else range(mean_att.shape[0]))
        
        # Highlight last row (predicting verb)
        ax.add_patch(plt.Rectangle((0, mean_att.shape[0]-1), mean_att.shape[1], 1, 
                                   fill=False, edgecolor='red', 
                                   linewidth=3, linestyle='--'))
        
        ax.set_title(f'{cond_label} (n={n_samples})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Attended Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.suptitle('Attention Patterns (Last Row = Predicting Verb)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_and_plot(model, test_dataloader, temperature=0.1, 
                     use_gumbel=True, save_dir=None, seed=42):
    """
    Complete analysis with separate accuracy and attention evaluation.
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # Run two-stage evaluation
    print("Evaluating model (accuracy + attention tracking)...")
    accuracies, attention_by_condition, sentence_details = eval_with_attention_analysis(
        model, test_dataloader, temperature, use_gumbel
    )
    
    # Compute attention statistics
    attention_stats = {}
    for condition, attn_matrices in attention_by_condition.items():
        if len(attn_matrices) > 0:
            stacked = np.stack(attn_matrices)
            mean_att = np.mean(stacked, axis=0)
            std_att = np.std(stacked, axis=0)
            attention_stats[condition] = (mean_att, std_att, attn_matrices)
        else:
            attention_stats[condition] = (None, None, [])
    
    # Print results
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)
    for condition in sorted(accuracies.keys()):
        print(f"{condition:25s}: {accuracies[condition]:.2%}")
    print("="*60)
    
    # Generate plots
    if save_dir:
        print(f"\nGenerating plots in {save_dir}...")
        
        # 1. Accuracy comparison
        plot_accuracy_comparison(
            accuracies, 
            save_path=f"{save_dir}/accuracy_comparison.png"
        )
        
        # 2. Attention grid with verb prediction highlighted
        plot_attention_grid_with_verb(
            attention_stats,
            test_dataloader.dataset,
            save_path=f"{save_dir}/attention_patterns_with_verb.png"
        )
        
        print(f"\n✓ All plots saved to {save_dir}/")
    
    return {
        'accuracies': accuracies,
        'attention_stats': attention_stats,
        'sentence_details': sentence_details
    }
    
import torch
import numpy as np
from collections import Counter

def diagnose_dataset(dataset, tokenizer):
    """Check for dataset encoding issues"""
    print("="*60)
    print("DATASET DIAGNOSTICS")
    print("="*60)
    
    # 1. Check for unknown tokens (encoded as 0)
    unknown_words = []
    zero_encoded = 0
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        
        # Check sentence words
        for word in item['sentence']:
            if tokenizer.stoi.get(word.lower(), -1) == 0:
                unknown_words.append(word.lower())
        
        # Check encoded sentence for zeros
        if 0 in item['encoded_sentence']:
            zero_encoded += 1
        
        # Check if correct/wrong verbs are encoded as 0
        if item['encoded_correct'] == 0:
            print(f"⚠️  Correct verb '{item['correct']}' encoded as 0 (unknown)")
        if item['encoded_wrong'] == 0:
            print(f"⚠️  Wrong verb '{item['wrong']}' encoded as 0 (unknown)")
    
    if unknown_words:
        print(f"\n⚠️  Found {len(unknown_words)} unknown word instances (encoded as 0)")
        print(f"   Most common: {Counter(unknown_words).most_common(10)}")
    
    if zero_encoded > 0:
        print(f"⚠️  {zero_encoded}/{len(dataset)} sentences contain token 0")
    
    # 2. Check sentence structure
    sample = dataset[0]
    print(f"\nSample sentence:")
    print(f"   Words: {sample['sentence']}")
    print(f"   Encoded: {sample['encoded_sentence']}")
    print(f"   Correct: {sample['correct']} -> {sample['encoded_correct']}")
    print(f"   Wrong: {sample['wrong']} -> {sample['encoded_wrong']}")
    
    # 3. Check condition distribution
    print(f"\nCondition distribution:")
    condition_counts = Counter([dataset[i]['condition'] for i in range(len(dataset))])
    for cond, count in sorted(condition_counts.items()):
        print(f"   {cond:25s}: {count:4d} samples")
    
    print("="*60 + "\n")


def diagnose_model_outputs(model, test_dataloader, device='cuda'):
    """Check if model outputs are reasonable"""
    print("="*60)
    print("MODEL OUTPUT DIAGNOSTICS")
    print("="*60)
    
    model.eval()
    is_transformer = hasattr(model, 'token_embedding')
    
    with torch.no_grad():
        # Get first batch
        batch = next(iter(test_dataloader))
        sentence = batch["encoded_sentence"].to(device)
        correct = batch["encoded_correct"].to(device)
        wrong = batch["encoded_wrong"].to(device)
        
        # Process first 5 words
        context = sentence[:, :5].transpose(0, 1)
        
        print(f"\nInput shape: {context.shape}")
        print(f"Input tokens (first sample): {context[:, 0].cpu().tolist()}")
        
        # Forward pass
        if is_transformer:
            out = model(context, temperature=1.0, use_gumbel=False)
        else:
            cache = model.init_cache(context)
            out, _ ,_,_= model.forward_with_attention_fixed(
                context, cache, temperature=1.0, use_gumbel=False, hard=False
            )
        
        print(f"Output shape: {out.shape}")
        
        # Check last position output (predicts position 5)
        last_logits = out[-1]  # [batch_size, vocab_size]
        
        print(f"\nLogits statistics (last position):")
        print(f"   Min: {last_logits.min().item():.2f}")
        print(f"   Max: {last_logits.max().item():.2f}")
        print(f"   Mean: {last_logits.mean().item():.2f}")
        print(f"   Std: {last_logits.std().item():.2f}")
        
        # Check if model is just outputting uniform distribution
        probs = torch.softmax(last_logits[0], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        
        print(f"\nTop 10 predictions (first sample):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"   {i+1}. Token {idx.item()}: {prob.item():.4f}")
        
        # Check correct vs wrong probabilities
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
        correct_lp = log_probs[0, correct[0]].item()
        wrong_lp = log_probs[0, wrong[0]].item()
        
        print(f"\nFirst sample comparison:")
        print(f"   Correct verb (token {correct[0].item()}): log_prob = {correct_lp:.4f}")
        print(f"   Wrong verb (token {wrong[0].item()}):   log_prob = {wrong_lp:.4f}")
        print(f"   Difference: {correct_lp - wrong_lp:.4f}")
        print(f"   Model prefers: {'CORRECT' if correct_lp >= wrong_lp else 'WRONG'} ❌" if correct_lp < wrong_lp else 'CORRECT ✓')
    
    print("="*60 + "\n")


def diagnose_attention_patterns(model, test_dataloader, device='cuda'):
    """Check if attention patterns make sense"""
    print("="*60)
    print("ATTENTION PATTERN DIAGNOSTICS")
    print("="*60)
    
    model.eval()
    is_transformer = hasattr(model, 'token_embedding')
    
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        sentence = batch["encoded_sentence"].to(device)
        context = sentence[:, :5].transpose(0, 1)
        
        # Get attention
        if is_transformer:
            _, attn_history = model.forward_with_attention_tracking(
                context, temperature=1.0, use_gumbel=False, hard=False
            )
        else:
            cache = model.init_cache(context)
            _, _, attn_history, _ = model.forward_with_attention_fixed(
                context, cache, temperature=1.0, use_gumbel=False, hard=False
            )
        
        print(f"\nAttention history length: {len(attn_history)}")
        
        # Check attention for last position (predicting verb)
        if len(attn_history) >= 5:
            last_attn = attn_history[4]  # Position 4 (predicting position 5)
            print(f"Last attention shape: {last_attn.shape}")
            
            # First sample, first head
            attn_weights = last_attn[0, 0, :].cpu().numpy()
            print(f"\nAttention weights at position 4 (predicting verb):")
            for i, weight in enumerate(attn_weights):
                print(f"   Position {i}: {weight:.4f}")
            
            print(f"\nAttention sum: {attn_weights.sum():.4f} (should be ~1.0)")
            
            # Check if attention is too uniform or too peaked
            entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-10))
            max_entropy = np.log(len(attn_weights))
            print(f"Attention entropy: {entropy:.4f} / {max_entropy:.4f} (higher = more uniform)")
            
            if entropy > 0.9 * max_entropy:
                print("⚠️  Attention is nearly uniform (not learning structure)")
            elif np.max(attn_weights) > 0.95:
                print("⚠️  Attention is too peaked (overfitting to one position)")
    
    print("="*60 + "\n")


def diagnose_verb_encoding(dataset, tokenizer):
    """Check if correct/wrong verbs are properly differentiated"""
    print("="*60)
    print("VERB ENCODING DIAGNOSTICS")
    print("="*60)
    
    correct_verbs = []
    wrong_verbs = []
    duplicate_encoding = 0
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        correct_verbs.append((item['correct'], item['encoded_correct']))
        wrong_verbs.append((item['wrong'], item['encoded_wrong']))
        
        if item['encoded_correct'] == item['encoded_wrong']:
            duplicate_encoding += 1
            print(f"⚠️  Sample {idx}: correct '{item['correct']}' and wrong '{item['wrong']}' "
                  f"have same encoding {item['encoded_correct']}")
    
    if duplicate_encoding > 0:
        print(f"\n⚠️  {duplicate_encoding} samples have identical encoding for correct/wrong verbs!")
    
    # Check unique verbs
    unique_correct = set([v for _, v in correct_verbs])
    unique_wrong = set([v for _, v in wrong_verbs])
    
    print(f"\nUnique correct verb tokens: {len(unique_correct)}")
    print(f"Unique wrong verb tokens: {len(unique_wrong)}")
    print(f"Overlap: {len(unique_correct & unique_wrong)}")
    
    print("="*60 + "\n")


def run_all_diagnostics(model, test_dataloader, tokenizer, device='cuda'):
    """Run all diagnostic tests"""
    model = model.to(device)
    
    print("\n" + "🔍 RUNNING FULL DIAGNOSTICS" + "\n")
    
    diagnose_dataset(test_dataloader.dataset, tokenizer)
    diagnose_verb_encoding(test_dataloader.dataset, tokenizer)
    diagnose_model_outputs(model, test_dataloader, device)
    diagnose_attention_patterns(model, test_dataloader, device)
    
    print("✅ Diagnostics complete!\n")