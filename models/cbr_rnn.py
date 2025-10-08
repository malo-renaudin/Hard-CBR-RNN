
import numpy as np
import torch.nn as nn
import torch
import gc
import torch.nn.functional as F
import math

         
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
            query = self.drop(F.relu(self.q_norm(self.q(torch.cat([curr_emb, curr_hidden], dim=-1)))))
            
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
            intermediate = self.drop(F.relu(self.int_norm(
                self.intermediate_h(torch.cat([curr_emb, query, attn, curr_hidden], dim=-1)))))
            
            final_output = self.drop(F.relu(self.f_norm(self.final_h(intermediate))))
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
            
class CBRLanguageModel(pl.LightningModule):
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
