import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        
    def scaled_dot_product_attention(self, Q, K, V, temperature=1.0, use_gumbel=False, hard=False, mask=None):
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
    
    def forward(self, query, key, value, temperature=1.0, use_gumbel=False, hard=False, mask=None):
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
            Q, K, V, mask, temperature=temperature, use_gumbel=use_gumbel, hard=hard
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
    
    def forward(self, x, temperature=1.0, use_gumbel=False, hard=False, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask, temperature=temperature, use_gumbel=use_gumbel, hard=hard)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class SimpleTransformer(nn.Module):
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


class SimpleTransformerLM(pl.LightningModule):
    """
    Lightning wrapper for Simple Transformer - matches your CBR interface
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, 
                 dropout=0.5, lr=3e-4, weight_decay=0.1,
                 use_gumbel_softmax=False, initial_temp=1.0, final_temp=0.1, temp_decay="exponential"):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize the transformer
        self.model = SimpleTransformer(
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

    
    

# # Usage - just replace your CBR model with this:
# def create_transformer_model(tokenizer):
#     """Create transformer model with similar capacity to your CBR model"""
#     return SimpleTransformerLM(
#         vocab_size=tokenizer.vocab_size,
#         d_model=512,      # Same as your nhid
#         nhead=8,          # 8 attention heads
#         num_layers=2,     # 2 transformer layers ≈ similar capacity
#         dropout=0.5,      # Same dropout
#         lr=3e-4,          # Same learning rate
#         weight_decay=0.1  # Same weight decay
#     )


# In your training script, just change:
# model = CBRLanguageModel(...) 
# to:
# model = create_transformer_model(tokenizer)