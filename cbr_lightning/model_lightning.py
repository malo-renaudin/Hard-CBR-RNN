# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from attention import MultiheadAttention
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from torchmetrics import MeanMetric


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
        self.save_hyperparameters()

        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=temperature,
            decay_rate=temp_decay_rate,
            final_temp=temp_final
        )
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax

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
        # Tying weights
        self.decoder.weight = self.encoder.weight
        # Adaptive pooling for cache compression
        # self.hidden_pool = nn.AdaptiveAvgPool1d(compressed_dim)
        # self.key_pool = nn.AdaptiveAvgPool1d(compressed_dim)
        # self.value_pool = nn.AdaptiveAvgPool1d(compressed_dim)
        # self.hidden_compress_norm = nn.LayerNorm(nhid)
        # self.key_compress_norm = nn.LayerNorm(nhid)
        # self.value_compress_norm = nn.LayerNorm(nhid)

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.epoch_cache = None

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
        bsz = observation.size(1) if len(observation.size()) > 1 else 1
        hidden = torch.zeros(self.compressed_dim, bsz,
                             self.nhid, device=self.device)
        key_cache = torch.zeros(bsz, self.compressed_dim,
                                self.nhid, device=self.device)
        value_cache = torch.zeros(
            bsz, self.compressed_dim, self.nhid, device=self.device)
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

    # def compress_cache(self, hidden, key_cache, value_cache):
    #     # hidden: [seq_len, batch, nhid] -> [compressed_dim, batch, nhid]
    #     h = self.hidden_pool(hidden.transpose(0,1).transpose(1,2)).transpose(1,2)
    #     hidden_compressed = self.drop(self.tanh(self.hidden_compress_norm(h))).transpose(0,1)

    #     # key_cache: [batch, seq_len, nhid] -> [batch, compressed_dim, nhid]
    #     k = self.key_pool(key_cache.transpose(1,2)).transpose(1,2)
    #     key_compressed = self.drop(self.tanh(self.key_compress_norm(k)))

    #     # value_cache: [batch, seq_len, nhid] -> [batch, compressed_dim, nhid]
    #     v = self.value_pool(value_cache.transpose(1,2)).transpose(1,2)
    #     value_compressed = self.drop(self.tanh(self.value_compress_norm(v)))

    #     return hidden_compressed, key_compressed, value_compressed

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
    # Training metrics computation
    # ----------------------
    def compute_vocabulary_metrics(self, observation):
        """Compute vocabulary-related metrics"""
        metrics = {}

        # Assuming UNK token has ID 0 (adjust based on your tokenizer)
        unk_token_id = 32  # Change this to your actual UNK token ID

        # Count UNK tokens
        total_tokens = observation.numel()
        unk_count = (observation == unk_token_id).sum().item()
        unk_percentage = (unk_count / total_tokens) * \
            100 if total_tokens > 0 else 0

        # Vocabulary coverage
        unique_tokens = torch.unique(observation).numel()

        metrics['unk_percentage'] = unk_percentage
        metrics['unique_tokens_in_batch'] = unique_tokens
        metrics['total_tokens_in_batch'] = total_tokens

        # Token distribution entropy
        token_counts = torch.bincount(observation.flatten())
        token_probs = token_counts.float() / token_counts.sum()
        token_probs = token_probs[token_probs > 0]  # Remove zeros
        token_entropy = -(token_probs * torch.log(token_probs)).sum().item()
        metrics['token_entropy'] = token_entropy

        return metrics

    def compute_gradient_metrics(self):
        """Compute gradient-related metrics"""
        total_norm = 0.0
        param_count = 0
        grad_count = 0
        layer_grad_norms = {}

        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_count += param.numel()

                # Layer-wise gradient norms
                layer_name = name.split('.')[0]
                if layer_name not in layer_grad_norms:
                    layer_grad_norms[layer_name] = 0.0
                layer_grad_norms[layer_name] += param_norm.item() ** 2

            param_count += param.numel()

        total_grad_norm = total_norm ** 0.5

        # Compute layer-wise norms
        for layer_name in layer_grad_norms:
            layer_grad_norms[layer_name] = layer_grad_norms[layer_name] ** 0.5

        return {
            'grad_norm': total_grad_norm,
            'grad_count': grad_count,
            'param_count': param_count,
            'layer_grad_norms': layer_grad_norms
        }

    def compute_weight_metrics(self):
        """Compute weight-related metrics"""
        total_weight_norm = 0.0
        layer_weight_norms = {}

        for name, param in self.named_parameters():
            param_norm = param.data.norm(2)
            total_weight_norm += param_norm.item() ** 2

            # Layer-wise weight norms
            layer_name = name.split('.')[0]
            if layer_name not in layer_weight_norms:
                layer_weight_norms[layer_name] = 0.0
            layer_weight_norms[layer_name] += param_norm.item() ** 2

        total_weight_norm = total_weight_norm ** 0.5

        # Compute layer-wise norms
        for layer_name in layer_weight_norms:
            layer_weight_norms[layer_name] = layer_weight_norms[layer_name] ** 0.5

        return {
            'weight_norm': total_weight_norm,
            'layer_weight_norms': layer_weight_norms
        }

    def compute_activation_metrics(self, hidden, key_cache, value_cache, attn_out):
        """Compute activation statistics"""
        metrics = {}

        # Hidden state statistics
        if hidden is not None:
            metrics['hidden_mean'] = hidden.mean().item()
            metrics['hidden_std'] = hidden.std().item()
            metrics['hidden_sparsity'] = (
                hidden.abs() < 1e-6).float().mean().item()

        # Attention output statistics
        if attn_out is not None:
            metrics['attn_mean'] = attn_out.mean().item()
            metrics['attn_std'] = attn_out.std().item()
            metrics['attn_sparsity'] = (
                attn_out.abs() < 1e-6).float().mean().item()

        return metrics

    def compute_learning_dynamics(self, loss):
        """Compute learning rate and update ratios"""
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Compute parameter update magnitudes vs parameter magnitudes
        update_ratios = {}
        if hasattr(self, '_prev_params'):
            for name, param in self.named_parameters():
                if name in self._prev_params:
                    update = param.data - self._prev_params[name]
                    update_norm = update.norm(2)
                    param_norm = param.data.norm(2)
                    if param_norm > 1e-8:
                        update_ratios[name] = (update_norm / param_norm).item()

        # Store current params for next iteration
        self._prev_params = {name: param.data.clone()
                             for name, param in self.named_parameters()}

        return {
            'learning_rate': current_lr,
            'update_ratios': update_ratios
        }

    # ----------------------
    # Forward
    # ----------------------
    def get_query(self, emb, hidden):
        combined = torch.cat((emb, hidden[-1]), -1)
        q = self.drop(self.tanh(self.q_norm(self.q(combined))))
        return q.unsqueeze(1)

    def intermediate_layers(self, i, emb, query, attn, hidden):
        inter_input = torch.cat((emb[i], query, attn, hidden[-1]), -1)
        inter = self.drop(self.tanh(self.int_norm(
            self.intermediate_h(inter_input))))
        final = self.drop(self.tanh(self.f_norm(self.final_h(inter))))
        k_i, v_i, h_i = final.split(self.nhid, dim=-1)
        return k_i, v_i, h_i

    def forward(self, observation, initial_cache=None):
        seq_len = observation.size(0)
        hidden, key_cache, value_cache = initial_cache if initial_cache else self.init_cache(
            observation)
        emb = self.drop(self.encoder(observation))

        for i in range(seq_len):
            query = self.get_query(emb[i], hidden)
            attn_out, _ = self.multihead_attn(
                query, key_cache, value_cache, self.temperature, self.gumbel_softmax, need_weights=False)
            attn_out = attn_out.squeeze(1)
            query = query.squeeze(1)
            k_i, v_i, h_i = self.intermediate_layers(
                i, emb, query, attn_out, hidden)
            key_cache, value_cache, hidden = self.update_cache(
                key_cache, value_cache, hidden, k_i, v_i, h_i)

        decoded = self.decoder(hidden[-self.seq_len:]).transpose(0, 1)
        cache = self.compress_cache(hidden, key_cache, value_cache)

        if self.training:
            self._last_hidden = hidden
            self._last_attn_out = attn_out
            self._last_key_cache = key_cache
            self._last_value_cache = value_cache

        return decoded, cache

    # ----------------------
    # Shared step for all stages
    # ----------------------
    def _shared_step(self, batch, stage):
        data, targets = batch
        vocab_metrics = self.compute_vocabulary_metrics(data)
        for k, v in vocab_metrics.items():
            self.log(f"{stage}_{k}", v, on_step=True,
                     on_epoch=True, prog_bar=False)

        data, targets = data.transpose(0, 1), targets.transpose(0, 1)
        if self.epoch_cache is None:
            self.epoch_cache = self.init_cache(data)

        output, new_cache = self.forward(data, initial_cache=self.epoch_cache)

        if new_cache is not None:
            self.epoch_cache = tuple(c.detach().clone() for c in new_cache)

        output_flat, targets_flat = output.reshape(
            -1, output.size(-1)), targets.reshape(-1)
        loss = F.cross_entropy(output_flat, targets_flat)
        ppl = torch.exp(loss)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)

        # Log detailed training metrics (only during training steps)
        if stage == "train" and self.training:
            # Gradient metrics (computed after backward pass)
            if hasattr(self, '_log_gradients') and self._log_gradients:
                grad_metrics = self.compute_gradient_metrics()
                self.log("train_grad_norm",
                         grad_metrics['grad_norm'], on_step=True, on_epoch=False)

                # Log some key layer gradient norms
                for layer_name, grad_norm in grad_metrics['layer_grad_norms'].items():
                    if layer_name in ['encoder', 'decoder', 'multihead_attn']:
                        self.log(
                            f"train_grad_norm_{layer_name}", grad_norm, on_step=True, on_epoch=False)

            # Weight metrics
            weight_metrics = self.compute_weight_metrics()
            self.log("train_weight_norm",
                     weight_metrics['weight_norm'], on_step=True, on_epoch=False)

            # Activation metrics
            if hasattr(self, '_last_hidden'):
                activation_metrics = self.compute_activation_metrics(
                    self._last_hidden, self._last_key_cache, self._last_value_cache, self._last_attn_out
                )
                for metric_name, value in activation_metrics.items():
                    self.log(f"train_{metric_name}", value,
                             on_step=True, on_epoch=False)

            # Learning dynamics
            learning_metrics = self.compute_learning_dynamics(loss)
            self.log("train_learning_rate",
                     learning_metrics['learning_rate'], on_step=True, on_epoch=False)
            self.log("train_temperature", self.temperature,
                     on_step=True, on_epoch=False)

            # Memory usage (if you want to track it)
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                self.log("train_gpu_memory_allocated",
                         memory_allocated, on_step=True, on_epoch=False)
                self.log("train_gpu_memory_reserved", memory_reserved,
                         on_step=True, on_epoch=False)

            # Cache compression ratio
            if hasattr(self, '_last_hidden'):
                compression_ratio = self.compressed_dim / \
                    self._last_hidden.size(
                        0) if self._last_hidden.size(0) > 0 else 1.0
                self.log("train_cache_compression_ratio",
                         compression_ratio, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(
        batch, "train")

    def validation_step(
        self, batch, batch_idx): return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx): return self._shared_step(
        batch, "test")

    def on_before_optimizer_step(self, optimizer):
        # Set flag to compute gradients on next training step
        self._log_gradients = True

    def on_after_optimizer_step(self, optimizer, optimizer_closure):
        # Reset flag after optimizer step
        self._log_gradients = False

    def on_train_epoch_end(self):
        self.epoch_cache = None

    def on_train_epoch_end(self): self.epoch_cache = None

    # ----------------------
    # Optimizers & schedulers
    # ----------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )

        # Much more conservative decay for plateau recovery
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.999  # Very slow decay: 0.999 instead of 0.995
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500  # Apply decay much less frequently
            }
        }

##########################################################################################################################
# TRANSFORMER
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
        self.attention = MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
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
        attn_out, _ = self.attention(
            x, x, x,  attn_mask=mask, temperature=temperature, gumbel_softmax=gumbel_softmax, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Transformer(pl.LightningModule):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, max_seq_len, dropout, temperature, gumbel_softmax, learning_rate, temp_decay_rate=0.95, temp_final=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        self.temp_decay_rate = temp_decay_rate
        self.temp_final = temp_final

        self.temp_scheduler = TemperatureScheduler(
            initial_temp=temperature,
            decay_rate=temp_decay_rate,
            final_temp=temp_final
        )

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size+1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size+1, bias=False)

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

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(
            seq_len, seq_len, device=device), diagonal=1)
        mask = mask.float()                 # ensure float
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_ids, temperature, gumbel_softmax):
        temperature = self.temperature
        gumbel_softmax = self.gumbel_softmax
        seq_len, batch_size = input_ids.size()
        input_ids = input_ids.transpose(0, 1)
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        # x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, input_ids.device)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask, temperature=temperature,
                      gumbel_softmax=gumbel_softmax)

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
        logits = self(input_ids, self.temperature, self.gumbel_softmax)

        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, targets = batch
        logits = self(input_ids, self.temperature, self.gumbel_softmax)

        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_targets = targets[..., 1:].contiguous()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        )

        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


##########################################################################################################################
# LSTM
##########################################################################################################################

class LSTM(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.2):
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
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, input_ids, hidden=None):
        seq_len, batch_size = input_ids.size()
        input_ids = input_ids.transpose(0, 1)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_ids.device)

        # Embedding
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # LSTM forward pass
        # (batch_size, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to vocabulary size
        # (batch_size, seq_len, vocab_size)
        logits = self.output_projection(lstm_out)
        logits = logits.transpose(0, 1)
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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
