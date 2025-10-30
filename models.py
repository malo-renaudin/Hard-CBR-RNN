
import numpy as np
import torch.nn as nn
import torch
import gc
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
         
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
        # Embedding
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.decoder.bias)

        # All linear layers — use Xavier (Glorot) for tanh
        for lin in [self.q, self.intermediate_h, self.final_h]:
            nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(lin.bias)

        # Layer norms
        for layer_norm in [self.q_norm, self.int_norm, self.f_norm]:
            if hasattr(layer_norm, 'weight'):
                nn.init.ones_(layer_norm.weight)
            if hasattr(layer_norm, 'bias'):
                nn.init.zeros_(layer_norm.bias)
            
            # Layer norm parameters (if they have learnable params)
            for layer_norm in [self.q_norm, self.int_norm, self.f_norm]:
                if hasattr(layer_norm, 'weight'):
                    nn.init.ones_(layer_norm.weight)
                if hasattr(layer_norm, 'bias'):
                    nn.init.zeros_(layer_norm.bias)
        
    def forward(self, observation, initial_cache, temperature=1.0, use_gumbel=False, hard=False):
        hidden_init, key_cache_init, value_cache_init = initial_cache
        seq_len, batch_size = observation.shape
        device = observation.device
        
        # Pre-allocate all state tensors
        states = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        keys = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)  
        values = torch.zeros(seq_len + 1, batch_size, self.nhid, device=device)
        
        # Initialize
        states[0] = hidden_init.squeeze(0)
        keys[0] = key_cache_init.squeeze(0)
        values[0] = value_cache_init.squeeze(0)
        
        emb = self.drop(self.encoder(observation))
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        
        for i in range(seq_len):
            cache_len = i + 1
            curr_emb, curr_hidden = emb[i], states[i]
            
            # Query computation
            query = self.drop(F.tanh(self.q_norm(self.q(torch.cat([curr_emb, curr_hidden], dim=-1)))))
            
            # Unified multi-head attention (single-head is just nheads=1)
            current_keys = keys[:cache_len].transpose(0, 1)  # batch_size x cache_len x nhid
            current_values = values[:cache_len].transpose(0, 1)  # batch_size x cache_len x nhid
            
            # Reshape for multi-head (works for single-head when nheads=1)
            q_mh = query.view(batch_size, self.nheads, self.head_dim)
            k_mh = current_keys.view(batch_size, cache_len, self.nheads, self.head_dim)
            v_mh = current_values.view(batch_size, cache_len, self.nheads, self.head_dim)
            
            # Compute attention scores: batch_size x nheads x cache_len
            attn_scores = torch.einsum('bnh,bcnh->bnc', q_mh, k_mh)
            
            # Apply mask and scaling
            masked_scores = (attn_scores + causal_mask[i, :cache_len].unsqueeze(0).unsqueeze(0)) / self.attn_div_factor
            
            # Softmax (Gumbel or regular)
            softmax_fn = (lambda x: F.gumbel_softmax(x, tau=temperature, hard=hard, dim=-1)) if use_gumbel else (lambda x: F.softmax(x, dim=-1))
            attn_weights = softmax_fn(masked_scores)
            
            # Compute attention output and flatten: batch_size x nhid
            attn = torch.einsum('bnc,bcnh->bnh', attn_weights, v_mh).contiguous().view(batch_size, self.nhid)
            
            # Process through network layers
            intermediate = self.drop(F.tanh(self.int_norm(
                self.intermediate_h(torch.cat([curr_emb, query, attn, curr_hidden], dim=-1)))))
            
            final_output = self.drop(F.tanh(self.f_norm(self.final_h(intermediate))))
            key_i, value_i, hidden_i = final_output.split(self.nhid, dim=-1)
            
            states[i + 1] = hidden_i
            keys[i + 1] = key_i
            values[i + 1] = value_i
        
        return self.decoder(states[1:]), states

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
            
class CBR_LM(pl.LightningModule):
    """PyTorch Lightning module for training CueBasedRNNModel with essential monitoring"""
    
    def __init__(self, vocab_size, ninp=512, nhid=512, nlayers=1, 
                 dropout=0.5, nheads=8, lr=1.0, weight_decay=0,
                 use_gumbel_softmax=False, initial_temp=1.0, final_temp=0.1, temp_decay="exponential"):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = CBR_RNN(
            ntoken=vocab_size,
            ninp=ninp,
            nhid=nhid, 
            nlayers=nlayers,
            dropout=dropout,
            nheads=nheads
        )
        
        # Add label smoothing to prevent overconfident predictions
        self.criterion = nn.CrossEntropyLoss()
        
        self.use_gumbel_softmax = use_gumbel_softmax
        self.initial_temp = initial_temp if use_gumbel_softmax else None
        self.final_temp = final_temp if use_gumbel_softmax else None
        self.temp_decay = temp_decay if use_gumbel_softmax else None
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        
    def get_current_temperature(self):
        if not self.use_gumbel_softmax:
            return 1.0
            
        if not hasattr(self.trainer, 'max_epochs') or self.trainer.max_epochs == 0:
            return self.initial_temp
            
        progress = self.current_epoch / self.trainer.max_epochs
        
        if self.temp_decay == "linear":
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.temp_decay == "exponential":
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.temp_decay == "cosine":
            temp = self.final_temp + (self.initial_temp - self.final_temp) * \
                   0.5 * (1 + math.cos(math.pi * progress))
        else:
            temp = self.initial_temp
            
        return max(temp, 0.1)
    
    def _shared_step(self, batch, stage):
        sequences, targets = batch
        
        # Basic input validation
        if sequences.min() < 0 or sequences.max() >= self.vocab_size:
            print(f"ERROR: Invalid token indices in {stage}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        sequences = sequences.transpose(0, 1)
        targets = targets.transpose(0, 1)
        
        initial_cache = self.model.init_cache(sequences)
        
        forward_kwargs = {
            'observation': sequences,
            'initial_cache': initial_cache
        }
        
        if self.use_gumbel_softmax:
            current_temp = self.get_current_temperature()
            forward_kwargs.update({
                'temperature': current_temp,
                'use_gumbel': True
            })
            self.log(f"{stage}_gumbel_temp", current_temp)
            
        # Forward pass
        try:
            output, final_hidden = self.model.forward(**forward_kwargs)
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Prepare for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Key metric: Logits range (model expressiveness)
        logit_max = output_flat.max(dim=-1)[0]
        logit_min = output_flat.min(dim=-1)[0]
        logit_range = (logit_max - logit_min).mean().item()
        self.log(f"{stage}_logit_range", logit_range)
        
        # Compute loss
        try:
            lm_loss = self.criterion(output_flat, targets_flat)
        except Exception as e:
            print(f"ERROR in loss computation: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Loss validation
        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            print(f"CRITICAL: Invalid loss detected: {lm_loss}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Basic prediction metrics
        predictions = output_flat.argmax(dim=-1)
        accuracy = (predictions == targets_flat).float().mean()
        
        # Prediction confidence and diversity metrics
        probs = F.softmax(output_flat, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        
        # Entropy (prediction diversity) - higher = more diverse
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Check for model collapse (predicting same token repeatedly)
        most_common_pred = torch.bincount(predictions.flatten()).max().item()
        repetition_ratio = most_common_pred / predictions.numel()
        
        # Compute perplexity
        ppl = torch.exp(torch.clamp(lm_loss, max=10))
        
        # Log core metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_accuracy", accuracy)
        self.log(f"{stage}_confidence", avg_confidence)  # How confident are predictions?
        self.log(f"{stage}_entropy_norm", normalized_entropy)  # How diverse are predictions?
        self.log(f"{stage}_repetition_ratio", repetition_ratio)  # Are we stuck on one token?
        
        return lm_loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        """Log weight norms at epoch end"""
        # Weight norms by layer
        weight_norms = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                norm = param.norm().item()
                weight_norms[name.replace('.', '_')] = norm
                self.log(f"weight_norm_{name.replace('.', '_')}", norm)
        
        # Overall weight statistics
        all_norms = list(weight_norms.values())
        if all_norms:
            self.log("weight_norm_max", max(all_norms))
            self.log("weight_norm_avg", sum(all_norms) / len(all_norms))
            
    def on_after_backward(self):
        """Log gradient norm after backward"""
        if self.global_step % 50 == 0:
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("grad_norm", total_norm, prog_bar=True)

    def on_train_epoch_start(self):
        """Log current learning rate"""
        for opt in self.trainer.optimizers:
            lr = opt.param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=True)
            
    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',           # monitor validation loss
            factor=0.5,           # reduce LR by half
            patience=2,           # wait 2 epochs without improvement
            min_lr=1e-5,          # minimum LR

        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # the metric to monitor
                "interval": "epoch",    # Reduce LR at epoch end
                "frequency": 1
            }
    }
class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid_dim=512, nlayers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hid_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.nlayers = nlayers
        self.hid_dim = hid_dim

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
        return (weight.new_zeros(self.nlayers, batch_size, self.hid_dim, device=device),
                weight.new_zeros(self.nlayers, batch_size, self.hid_dim, device=device))


class LSTM_LM(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim, hid_dim, nlayers, dropout, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = LSTM(vocab_size=vocab_size,
                                emb_dim=emb_dim,
                                hid_dim=hid_dim,
                                nlayers=nlayers,
                                dropout=dropout)
        self.hidden = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
    def _shared_step(self, batch, stage):
        """Shared step for train/val/test"""
        sequences, targets = batch
        batch_size, seq_len = sequences.shape
        device = sequences.device
        if sequences.min() < 0 or sequences.max() >= self.vocab_size:
            print(f"ERROR: Invalid token indices in {stage}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        if stage == 'train':
            self.hidden = self.model.init_hidden(batch_size, device)
            
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
            # Forward pass
            try:
                output, self.hidden = self.model(sequences, self.hidden)

            except Exception as e:
                print(f"ERROR in forward pass: {e}")
                return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
            
        elif stage == 'val':
            self.hidden = self.model.init_hidden(batch_size, device)
            output, self.hidden = self.model(sequences, self.hidden)
            
        # Prepare for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Key metric: Logits range (model expressiveness)
        logit_max = output_flat.max(dim=-1)[0]
        logit_min = output_flat.min(dim=-1)[0]
        logit_range = (logit_max - logit_min).mean().item()
        self.log(f"{stage}_logit_range", logit_range)
        
        # Compute loss
        try:
            lm_loss = self.criterion(output_flat, targets_flat)
        except Exception as e:
            print(f"ERROR in loss computation: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Loss validation
        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            print(f"CRITICAL: Invalid loss detected: {lm_loss}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Basic prediction metrics
        predictions = output_flat.argmax(dim=-1)
        accuracy = (predictions == targets_flat).float().mean()
        
        # Prediction confidence and diversity metrics
        probs = F.softmax(output_flat, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        
        # Entropy (prediction diversity) - higher = more diverse
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Check for model collapse (predicting same token repeatedly)
        most_common_pred = torch.bincount(predictions.flatten()).max().item()
        repetition_ratio = most_common_pred / predictions.numel()
        
        # Compute perplexity
        ppl = torch.exp(torch.clamp(lm_loss, max=10))
        
        # Log core metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_accuracy", accuracy)
        self.log(f"{stage}_confidence", avg_confidence)  # How confident are predictions?
        self.log(f"{stage}_entropy_norm", normalized_entropy)  # How diverse are predictions?
        self.log(f"{stage}_repetition_ratio", repetition_ratio)  # Are we stuck on one token?
        
        return lm_loss
    
    
    def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        """Log weight norms at epoch end"""
        # Weight norms by layer
        weight_norms = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                norm = param.norm().item()
                weight_norms[name.replace('.', '_')] = norm
                self.log(f"weight_norm_{name.replace('.', '_')}", norm)
        
        # Overall weight statistics
        all_norms = list(weight_norms.values())
        if all_norms:
            self.log("weight_norm_max", max(all_norms))
            self.log("weight_norm_avg", sum(all_norms) / len(all_norms))
            
    def on_after_backward(self):
        """Log gradient norm after backward"""
        if self.global_step % 50 == 0:
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("grad_norm", total_norm, prog_bar=True)

    def on_train_epoch_start(self):
        """Log current learning rate"""
        for opt in self.trainer.optimizers:
            lr = opt.param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=True)
    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',           # monitor validation loss
            factor=0.5,           # reduce LR by half
            patience=2,           # wait 2 epochs without improvement
            min_lr=1e-5,          # minimum LR

        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # the metric to monitor
                "interval": "epoch",    # Reduce LR at epoch end
                "frequency": 1
            }
        }

class MultiHeadAttention(nn.Module):
    """
    Classic Multi-Head Attention implementation
    Exactly as described in "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, temperature=1.0, use_gumbel=False, hard=False):
        """
        Classic scaled dot-product attention
        Q, K, V: [batch_size, n_heads, seq_len, d_k]
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal/padding masks)
        if mask is not None:
            # Expand mask to match scores dimensions
            # mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        softmax_fn = (lambda x: F.gumbel_softmax(x, tau=temperature, hard=hard, dim=-1)) if use_gumbel else (lambda x: F.softmax(x, dim=-1))

        attention_weights = softmax_fn(scores)

        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None, temperature=1.0, use_gumbel=False, hard=False):
        """
        Forward pass
        query, key, value: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len] or broadcastable
        """
        batch_size, seq_len, d_model = query.shape
        
        # 1. Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model] 
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # 2. Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q=Q, K=K, V=V, mask=mask, temperature=temperature, use_gumbel=use_gumbel, hard=hard
        )
        
        # 4. Concatenate heads
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights


class TransformerLayer(nn.Module):
    """
    Single Transformer layer with our custom attention
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, temperature=1.0, use_gumbel=False, hard=False):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask=mask, temperature=temperature, use_gumbel=use_gumbel, hard=hard)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class Transformer(nn.Module):
    """
    Simple Transformer using our custom MultiHeadAttention
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(64, d_model)  # Max sequence length
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
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
        # src: [seq_len, batch_size] -> transpose to [batch_size, seq_len]
        src = src.transpose(0, 1)
        batch_size, seq_len = src.shape
        device = src.device
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(src) * math.sqrt(self.d_model)  # Scale embeddings
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Expand for batch
        
        # Pass through transformer layers
        for layer in self.layers:
            x, _ = layer(x=x, mask = causal_mask, temperature=temperature, use_gumbel=use_gumbel, hard=hard)
        
        # Project to vocabulary and transpose back to [seq_len, batch_size, vocab_size]
        logits = self.output_projection(x).transpose(0, 1)
        
        return logits


class Transformer_LM(pl.LightningModule):
    """
    Lightning wrapper for Simple Transformer - matches your CBR interface
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, 
                 dropout=0.5, lr=3e-4, weight_decay=0.1,
                 use_gumbel_softmax=False, initial_temp=1.0, final_temp=0.1, temp_decay="exponential"):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize the transformer
        self.model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
        self.use_gumbel_softmax = use_gumbel_softmax
        self.initial_temp = initial_temp if use_gumbel_softmax else None
        self.final_temp = final_temp if use_gumbel_softmax else None
        self.temp_decay = temp_decay if use_gumbel_softmax else None
    
    def get_current_temperature(self):
        if not self.use_gumbel_softmax:
            return 1.0
            
        if not hasattr(self.trainer, 'max_epochs') or self.trainer.max_epochs == 0:
            return self.initial_temp
            
        progress = self.current_epoch / self.trainer.max_epochs
        
        if self.temp_decay == "linear":
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.temp_decay == "exponential":
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.temp_decay == "cosine":
            temp = self.final_temp + (self.initial_temp - self.final_temp) * \
                   0.5 * (1 + math.cos(math.pi * progress))
        else:
            temp = self.initial_temp
            
        return max(temp, 0.1)
    
    def _shared_step(self, batch, stage):
        """Shared step for train/val/test"""
        sequences, targets = batch
        batch_size, seq_len = sequences.shape
        
        if sequences.min() < 0 or sequences.max() >= self.vocab_size:
            print(f"ERROR: Invalid token indices in {stage}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Transpose to match transformer expectation: [seq_len, batch_size]
        sequences = sequences.transpose(0, 1)
        targets = targets.transpose(0, 1)
        
        forward_kwargs = {
            'src': sequences,
        }
        
        if self.use_gumbel_softmax:
            current_temp = self.get_current_temperature()
            forward_kwargs.update({
                'temperature': current_temp,
                'use_gumbel': True
            })
            self.log(f"{stage}_gumbel_temp", current_temp)
        # Forward pass
        try:
            output = self.model.forward(**forward_kwargs)
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Prepare for loss computation
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        # Key metric: Logits range (model expressiveness)
        logit_max = output_flat.max(dim=-1)[0]
        logit_min = output_flat.min(dim=-1)[0]
        logit_range = (logit_max - logit_min).mean().item()
        self.log(f"{stage}_logit_range", logit_range)
        
        # Compute loss
        try:
            lm_loss = self.criterion(output_flat, targets_flat)
        except Exception as e:
            print(f"ERROR in loss computation: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Loss validation
        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            print(f"CRITICAL: Invalid loss detected: {lm_loss}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Basic prediction metrics
        predictions = output_flat.argmax(dim=-1)
        accuracy = (predictions == targets_flat).float().mean()
        
        # Prediction confidence and diversity metrics
        probs = F.softmax(output_flat, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        
        # Entropy (prediction diversity) - higher = more diverse
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Check for model collapse (predicting same token repeatedly)
        most_common_pred = torch.bincount(predictions.flatten()).max().item()
        repetition_ratio = most_common_pred / predictions.numel()
        
        # Compute perplexity
        ppl = torch.exp(torch.clamp(lm_loss, max=10))
        
        # Log core metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_accuracy", accuracy)
        self.log(f"{stage}_confidence", avg_confidence)  # How confident are predictions?
        self.log(f"{stage}_entropy_norm", normalized_entropy)  # How diverse are predictions?
        self.log(f"{stage}_repetition_ratio", repetition_ratio)  # Are we stuck on one token?
        
        return lm_loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        """Log weight norms at epoch end"""
        # Weight norms by layer
        weight_norms = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                norm = param.norm().item()
                weight_norms[name.replace('.', '_')] = norm
                self.log(f"weight_norm_{name.replace('.', '_')}", norm)
        
        # Overall weight statistics
        all_norms = list(weight_norms.values())
        if all_norms:
            self.log("weight_norm_max", max(all_norms))
            self.log("weight_norm_avg", sum(all_norms) / len(all_norms))
            
    def on_after_backward(self):
        """Log gradient norm after backward"""
        if self.global_step % 50 == 0:
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("grad_norm", total_norm, prog_bar=True)

    def on_train_epoch_start(self):
        """Log current learning rate"""
        for opt in self.trainer.optimizers:
            lr = opt.param_groups[0]["lr"]
            self.log("learning_rate", lr, prog_bar=True)
    def configure_optimizers(self):
        # --- Optimizer ---
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # --- Warmup Scheduler ---
        def lr_lambda(epoch):
            warmup_epochs = 5  # number of warmup epochs
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # linearly increase
            return 1.0  # keep LR at base level after warmup

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # --- Plateau Scheduler (used after warmup) ---
        reduce_on_plateau = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        # --- Combine both ---
        return {
            "optimizer": optimizer,
            "lr_scheduler": [
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "warmup_lr",
                },
                {
                    "scheduler": reduce_on_plateau,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "reduce_on_plateau",
                },
            ],
        }