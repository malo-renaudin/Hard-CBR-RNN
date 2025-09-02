# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from attention import MultiheadAttention

class TemperatureScheduler:
    """Simple exponential decay temperature scheduler"""
    
    def __init__(self, initial_temp=1.0, decay_rate=0.95, final_temp=0.1):
        self.initial_temp = initial_temp
        self.decay_rate = decay_rate
        self.final_temp = final_temp
        self.current_epoch = 0
    
    def step(self):
        """Update the current epoch"""
        self.current_epoch += 1
    
    def get_temperature(self):
        """Get current temperature using exponential decay"""
        temp = self.initial_temp * (self.decay_rate ** self.current_epoch)
        return max(temp, self.final_temp)
    
    
class CBR_RNN(pl.LightningModule):
    def __init__(self, ntoken, ninp, nhid, nheads, seq_len, compressed_dim, dropout=0.5, learning_rate=1e-3, 
                 temperature=1.0, gumbel_softmax=False, criterion='cross_entropy',
                 optimizer_type='adam', weight_decay=0.0, scheduler_type=None, temp_decay_rate=0.95, temp_final=0.1):
        super().__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=temperature,
            decay_rate=temp_decay_rate,
            final_temp=temp_final
        )
        
        # Store initial temperature
        self.initial_temperature = temperature
        self.epoch_cache = None
        self.seq_len = seq_len
        self.compressed_dim = compressed_dim
        self.nheads = nheads
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        self.score_attn = nn.Softmax(dim=-1)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.q = nn.Linear(ninp + nhid, nhid)
        self.intermediate_h = nn.Linear(nhid * 3 + ninp, nhid * 4)
        self.decoder = nn.Linear(nhid, ntoken)
        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)
        self.nhid = nhid
        self.final_h = nn.Linear(nhid * 4, nhid * 3)
        self.multihead_attn = MultiheadAttention(
            embed_dim=nhid, num_heads=nheads, batch_first=True
        )
        self.hidden_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim)
        self.key_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim) 
        self.value_compress = nn.Linear(nhid*(seq_len+compressed_dim), nhid*compressed_dim)
        self.hidden_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        self.key_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        self.value_compress_norm = nn.LayerNorm(nhid * compressed_dim)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        self.init_weights()

    def init_weights(self):
        """Initialize model weights for better training dynamics"""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param)
                elif "encoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif "decoder" in name:
                    nn.init.normal_(param, mean=0, std=0.01)
                elif "compress" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="tanh")
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_cache(self, observation):
        """Initialize hidden state and attention caches"""
        if len(observation.size()) > 1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1

        hidden = torch.zeros(self.compressed_dim, bsz, self.nhid).to(self.device) 
        key_cache = torch.zeros(bsz, self.compressed_dim, self.nhid).to(self.device) 
        value_cache = torch.zeros(bsz, self.compressed_dim, self.nhid).to(self.device) 
        return hidden, key_cache, value_cache

    def update_cache(self, key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i):
        hidden_i = hidden_i.unsqueeze(0)
        hidden = torch.cat((hidden, hidden_i), dim=0)
        key_cache_i = key_cache_i.unsqueeze(1)
        value_cache_i = value_cache_i.unsqueeze(1)
        key_cache = torch.cat((key_cache, key_cache_i), dim=1)
        value_cache = torch.cat((value_cache, value_cache_i), dim=1)
            
        return key_cache, value_cache, hidden

    def compress_cache(self, hidden, key_cache, value_cache):
        """
        Learned projection from [bsz, seq_len, nhid] to [bsz, compressed_dim, nhid]
        """
        # For hidden: [seq_len, batch, nhid] -> [batch, seq_len, nhid] -> [batch, compressed_dim, nhid] -> [compressed_dim, batch, nhid]
        hidden_reshaped = hidden.transpose(0, 1)  # [batch, seq_len, nhid]
        batch_size, seq_len, nhid = hidden_reshaped.shape

        hidden_flat = hidden_reshaped .reshape(batch_size, -1) 
        hidden_proj = self.drop(self.tanh(self.hidden_compress_norm(self.hidden_compress(hidden_flat))))  # [batch, nhid * compressed_dim]
        hidden_compressed = hidden_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        hidden_compressed = hidden_compressed.transpose(0, 1)  # [compressed_dim, batch, nhid]
        
        
        key_flat = key_cache.reshape(batch_size, -1)  # [batch, seq_len * nhid]
        key_proj = self.drop(self.tanh(self.key_compress_norm(self.key_compress(key_flat))))  # [batch, nhid * compressed_dim]
        key_compressed = key_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        
        
        value_flat = value_cache.reshape(batch_size, -1)  # [batch, seq_len * nhid]
        value_proj = self.drop(self.tanh(self.value_compress_norm(self.value_compress(value_flat))))  # [batch, nhid * compressed_dim]
        value_compressed = value_proj.reshape(batch_size, self.compressed_dim, nhid)  # [batch, compressed_dim, nhid]
        
        return hidden_compressed, key_compressed, value_compressed
    

    def intermediate_layers(self, i, emb, query, attn, hidden):

        intermediate_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)
        intermediate = self.drop(
            self.tanh(self.int_norm(self.intermediate_h(intermediate_input)))
        )
        final_output = self.drop(self.tanh(self.f_norm(self.final_h(intermediate))))
        key_cache_i, value_cache_i, hidden_i = final_output.split(self.nhid, dim=-1)
        return key_cache_i, value_cache_i, hidden_i

    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        query = self.drop(self.tanh(self.q_norm(self.q(combined))))
        query = query.unsqueeze(1)
        return query
    


    def forward(self, observation, initial_cache=None, nheads=None, temperature=None, gumbel_softmax=None):
        # Use instance variables if not provided
        nheads = nheads if nheads is not None else self.nheads
        temperature = temperature if temperature is not None else self.temperature
        gumbel_softmax = gumbel_softmax if gumbel_softmax is not None else self.gumbel_softmax
        
        seq_len = observation.size(0)
    
        if initial_cache is not None:
            hidden, key_cache, value_cache = initial_cache
        else:
            # Fallback to fresh cache if none provided
            hidden, key_cache, value_cache = self.init_cache(observation)
            

        emb = self.drop(self.encoder(observation))
        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            attn_output,_= self.multihead_attn(query, key_cache, value_cache, temperature, gumbel_softmax, need_weights=False)
            attn_output, query=attn_output.squeeze(1), query.squeeze(1)
            key_cache_i, value_cache_i, hidden_i = self.intermediate_layers(i, emb, query, attn_output, hidden)
            key_cache, value_cache, hidden = self.update_cache(key_cache, value_cache, hidden, key_cache_i, value_cache_i, hidden_i)
        # decoded = self.decoder(hidden[1:])
        decoded = self.decoder(hidden[:self.seq_len])
        cache = self.compress_cache(hidden, key_cache, value_cache)
        return decoded, cache
    
    def on_train_epoch_start(self):
        """Update temperature at the start of each training epoch"""
        if self.gumbel_softmax:
           self.temp_scheduler.step()
           self.temperature = self.temp_scheduler.get_temperature()
            
    def training_step(self, batch, batch_idx):
        # Extract data and targets from batch
        data, targets = batch
        # Initialize cache once per epoch or use existing epoch cache
        if self.epoch_cache is None:
            # Initialize cache once per epoch
            self.epoch_cache = self.init_cache(data)
        else:
            # Detach from computational graph but keep values
            hidden, key_cache, value_cache = self.epoch_cache
            self.epoch_cache = (
                hidden.detach().clone(),
                key_cache.detach().clone(), 
                value_cache.detach().clone()
            )
        
        
        # Forward pass
        output, new_cache = self.forward(
            data, 
            initial_cache= self.epoch_cache, 
            nheads=self.nheads, 
            temperature=self.temperature, 
            gumbel_softmax=self.gumbel_softmax
        )
        
        if new_cache is not None:
            self.epoch_cache = tuple(c.detach() for c in new_cache)
            
        
        # Reshape outputs and targets for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        if self.criterion == 'cross_entropy':
            loss = F.cross_entropy(output_flat, targets_flat)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        
        # Calculate perplexity for logging
        ppl = torch.exp(loss)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_ppl', ppl, prog_bar=True, on_step=True, on_epoch=True)
        self.log('temperature', self.temperature, on_step=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        # Extract data and targets from batch
        data, targets = batch
        
        # Initialize cache for CBR_RNN
        cache = self.init_cache(data)
        
        # Forward pass
        output, _ = self.forward(
            data, 
            initial_cache=cache, 
            nheads=self.nheads, 
            temperature=self.temperature, 
            gumbel_softmax=self.gumbel_softmax
        )
        
        # Reshape outputs and targets for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        if self.criterion == 'cross_entropy':
            loss = F.cross_entropy(output_flat, targets_flat)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
        
        # Calculate perplexity for logging
        ppl = torch.exp(loss)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ppl', ppl, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
##########################################################################################################################
#TRANSFORMER
##########################################################################################################################

class TemperatureScheduler:
    """Simple exponential decay temperature scheduler"""
    
    def __init__(self, initial_temp=1.0, decay_rate=0.95, final_temp=0.1):
        self.initial_temp = initial_temp
        self.decay_rate = decay_rate
        self.final_temp = final_temp
        self.current_epoch = 0
    
    def step(self):
        """Update the current epoch"""
        self.current_epoch += 1
    
    def get_temperature(self):
        """Get current temperature using exponential decay"""
        temp = self.initial_temp * (self.decay_rate ** self.current_epoch)
        return max(temp, self.final_temp)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, temperature, gumbel_softmax, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x,  attn_mask=mask, temperature=temperature, gumbel_softmax=gumbel_softmax, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Transformer(pl.LightningModule):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, 
                 d_ff, max_seq_len, dropout, lr, temperature, gumbel_softmax):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, input_ids):
        seq_len = input_ids.size(0)
        print('seq_len', seq_len)
        input_ids=input_ids.transpose(0,1)
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        # x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask, self.temperature, self.gumbel_softmax)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    
    def on_train_epoch_start(self):
        """Update temperature at the start of each training epoch"""
        self.temp_scheduler.step()
        self.temperature = self.temp_scheduler.get_temperature()
            
            
    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids, self.temperature, self.gumbel_softmax)
        
        # Shift logits and targets for next token prediction
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids)
        
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    

##########################################################################################################################
#LSTM
##########################################################################################################################

class LSTM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, input_ids, hidden=None):
        seq_len, batch_size= input_ids.size()
        input_ids = input_ids.transpose(0,1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_ids.device)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary size
        logits = self.output_projection(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def training_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits, _ = self(input_ids)
        
        # Shift logits and targets for next token prediction
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits, _ = self(input_ids)
        
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )
        
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    
    
