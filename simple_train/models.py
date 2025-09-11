import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
from pathlib import Path
from collections import Counter
import math
from attention import MultiheadAttention
from transformer import TransformerEncoderLayer
from enum import Enum



class CBR_RNN(nn.Module):
    def __init__(self, ntoken, ninp=512, nhid=512, nheads=1, seq_len=128, compressed_dim=1, dropout=0.5,
                 attention_type="standard"):
        super().__init__()

        # Model components
        self.encoder = nn.Embedding(ntoken+1, ninp)
        self.nhid = nhid
        self.seq_len = seq_len
        self.compressed_dim = compressed_dim
        self.nheads = nheads
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.q = nn.Linear(ninp + nhid, nhid)
        self.q_norm = nn.LayerNorm(nhid)

        self.intermediate_h = nn.Linear(nhid*3 + ninp, nhid*4)
        self.int_norm = nn.LayerNorm(nhid*4)
        self.f_norm = nn.LayerNorm(nhid*3)
        self.final_h = nn.Linear(nhid*4, nhid*3)

        self.decoder = nn.Linear(nhid, ntoken+1)
        self.multihead_attn = MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True)

        self.attention_type = attention_type  # ← Add this
        self.current_temperature = 1.0 
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name or "decoder" in name:
                    nn.init.normal_(param, 0, 0.1)
                elif "multihead_attn" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(
                        param, mode="fan_in", nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    # ----------------------
    # Cache handling
    # ----------------------

    def init_cache(self, observation):
        device = observation.device
        bsz = observation.size(0) #if len(observation.size()) > 1 else 1
        hidden = torch.zeros(self.compressed_dim, bsz,
                             self.nhid, device = device)
        key_cache = torch.zeros(bsz, self.compressed_dim,
                                self.nhid, device=device)
        value_cache = torch.zeros(
            bsz, self.compressed_dim, self.nhid, device=device)
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_i, value_i, hidden_i):
        # Ensure old cache has NO graph connections
        hidden_detached = hidden.detach() if hidden.requires_grad else hidden
        key_detached = key_cache.detach() if key_cache.requires_grad else key_cache
        value_detached = value_cache.detach() if value_cache.requires_grad else value_cache

        # Now safe to concatenate
        hidden = torch.cat((hidden_detached, hidden_i.unsqueeze(0)), dim=0)
        key_cache = torch.cat((key_detached, key_i.unsqueeze(1)), dim=1)
        value_cache = torch.cat((value_detached, value_i.unsqueeze(1)), dim=1)
        return key_cache, value_cache, hidden


    def compress_cache(self, hidden, key_cache, value_cache):
        # Keep only the last N tokens (most recent)
        if hidden.size(0) > self.compressed_dim:
            hidden_compressed = hidden[-self.compressed_dim:]
            key_compressed = key_cache[:, -self.compressed_dim:]
            value_compressed = value_cache[:, -self.compressed_dim:]
        else:
            hidden_compressed = hidden
            key_compressed = key_cache
            value_compressed = value_cache

        return hidden_compressed, key_compressed, value_compressed


    # ----------------------
    # Forward
    # ----------------------
    def _call_attention(self, query, key_cache, value_cache):
        """Clean attention dispatch using match/case"""
        if self.attention_type=='standard':
                return self.multihead_attn(
                    query, key_cache, value_cache, 
                    need_weights=False,
                    gumbel_softmax=False,
                    temperature=1.0  # Ignored for standard
                )
            
        elif self.attention_type =='gumbel':
                return self.multihead_attn(
                    query, key_cache, value_cache, 
                    need_weights=False,
                    gumbel_softmax=True,
                    temperature=self.current_temperature
                )
        else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
                
    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        q = self.drop(self.tanh(self.q_norm(self.q(combined))))
        return q.unsqueeze(1)

    def intermediate_layers(self, i, emb, query, attn, hidden):
        inter_input = torch.cat((emb[:,i,:], query, attn, hidden[-1]), -1)
        inter = self.drop(self.tanh(self.int_norm(
            self.intermediate_h(inter_input))))
        final = self.drop(self.tanh(self.f_norm(self.final_h(inter))))
        k_i, v_i, h_i = final.split(self.nhid, dim=-1)
        return k_i, v_i, h_i

    def forward(self, observation, initial_cache=None):
        seq_len = observation.size(1)
        hidden, key_cache, value_cache = initial_cache if initial_cache else self.init_cache(
            observation)
        emb = self.drop(self.encoder(observation))
        for i in range(seq_len):
            query = self.get_query(emb[:,i,:], hidden)
            attn_out, _ = self._call_attention(query, key_cache, value_cache)
            attn_out = attn_out.squeeze(1)
            query = query.squeeze(1)
            k_i, v_i, h_i = self.intermediate_layers(
                i, emb, query, attn_out, hidden)
            key_cache, value_cache, hidden = self.update_cache(
                key_cache, value_cache, hidden, k_i, v_i, h_i)
        decoded = self.decoder(hidden[-self.seq_len:])#.transpose(0, 1)
        cache = self.compress_cache(hidden, key_cache, value_cache)

        return decoded, cache

class LSTM(nn.Module):
    def __init__(self, vocab_size, ninp=512, nhid=512, nlayers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.nlayers = nlayers
        self.nhid = nhid

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid, device=device),
                weight.new_zeros(self.nlayers, batch_size, self.nhid, device=device))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nheads=8, nlayers=6, d_ff=2048, seq_len=512, dropout=0.1,
                 attention_type='standard'):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=d_ff, dropout=dropout, 
                                                batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.attention_type = attention_type
        self.current_temperature = 1.0
        
    def _get_attention_params(self):
        """Get attention parameters using match/case"""
        if self.attention_type=='standard':
                return {
                    "gumbel_softmax": False,
                    "temperature": 1.0
                }
        elif self.attention_type=='gumbel':
                return {
                    "gumbel_softmax": True,
                    "temperature": self.current_temperature
                }
        else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.pos_encoding(x)
        # Causal mask for language modeling
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        attn_params = self._get_attention_params()
        
        # Pass through transformer with appropriate parameters
        x = self.transformer(x, mask=mask, **attn_params)
        
        x = self.ln(x)
        logits = self.head(x)
        return logits  # shape: (batch, seq_len, vocab_size)
