import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl

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
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
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
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
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
            Q, K, V, mask
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
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
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
        self.pos_embedding = nn.Embedding(1000, d_model)  # Max sequence length
        
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
    
    def forward(self, src):
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
            x, _ = layer(x, causal_mask)
        
        # Project to vocabulary and transpose back to [seq_len, batch_size, vocab_size]
        logits = self.output_projection(x).transpose(0, 1)
        
        return logits


class SimpleTransformerLM(pl.LightningModule):
    """
    Lightning wrapper for Simple Transformer - matches your CBR interface
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, 
                 dropout=0.5, lr=3e-4, weight_decay=0.1):
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
    def _shared_step(self, batch, stage):
        """Shared step for train/val/test"""
        sequences, targets = batch
        batch_size, seq_len = sequences.shape
        
        # Transpose to match transformer expectation: [seq_len, batch_size]
        sequences = sequences.transpose(0, 1)
        targets = targets.transpose(0, 1)
        
        # Forward pass
        logits = self.model(sequences)
        
        # Compute loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = self.criterion(logits_flat, targets_flat)
        ppl = torch.exp(loss)
        
        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), 
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def configure_optimizers(self):
        """Same optimizer as your CBR model for fair comparison"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


# Usage - just replace your CBR model with this:
def create_transformer_model(tokenizer):
    """Create transformer model with similar capacity to your CBR model"""
    return SimpleTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=512,      # Same as your nhid
        nhead=8,          # 8 attention heads
        num_layers=2,     # 2 transformer layers ≈ similar capacity
        dropout=0.5,      # Same dropout
        lr=3e-4,          # Same learning rate
        weight_decay=0.1  # Same weight decay
    )


# In your training script, just change:
# model = CBRLanguageModel(...) 
# to:
# model = create_transformer_model(tokenizer)