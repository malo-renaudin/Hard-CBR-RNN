from models import CBR_RNN, Transformer, LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
from pathlib import Path
from collections import Counter
from cbr_lightning.attention import MultiheadAttention
import math
from enum import Enum 

class AttentionType(Enum):
    STANDARD = "standard"
    GUMBEL = "gumbel"



class CBR_RNN_Lightning(pl.LightningModule):
    def __init__(self, vocab_size, nhid, ninp, nheads, seq_len, compressed_dim, dropout, learning_rate, weight_decay):
        super().__init__()
        self.model = CBR_RNN(ntoken=vocab_size, nhid=nhid, 
                             ninp=ninp, nheads=nheads, seq_len=seq_len, 
                             compressed_dim=compressed_dim, dropout=dropout, 
                             learning_rate=learning_rate, weight_decay=weight_decay)
        self.epoch_cache = None
        learning_rate=self.learning_rate
        weight_decay=self.weight_decay
        
    def _shared_step(self, batch, stage):
        data, targets = batch
        if self.epoch_cache is None:
            self.epoch_cache = self.model.init_cache(data)

        output, new_cache = self.model.forward(data, initial_cache=self.epoch_cache)

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

        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(
        batch, "train")

    def validation_step(
        self, batch, batch_idx): return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx): return self._shared_step(
        batch, "test")

    def on_train_epoch_end(self):
        self.epoch_cache = None
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,#2e-4,
            weight_decay=self.weight_decay,#0.1,
            eps=1e-6,
            betas=(0.9, 0.95)
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.15 * total_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm"
        }


class LSTM_Lightning(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = LSTM(vocab_size)
        self.hidden = None

    def on_train_epoch_end(self):
        self.hidden = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        device = x.device

        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.hidden = self.model.init_hidden(batch_size, device)

        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        logits, self.hidden = self.model(x, self.hidden)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        device = x.device
        hidden = self.model.init_hidden(batch_size, device)
        logits, _ = self.model(x, hidden)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=-100)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=20.0)

class TransformerLightning(pl.LightningModule):
    def __init__(self, vocab_size, 
                 attention_type=AttentionType.STANDARD,
                 # Temperature scheduling parameters
                 initial_temperature=2.0,
                 final_temperature=0.1,
                 temperature_decay_steps=20000,
                 # Model parameters
                 d_model=512, nheads=8, nlayers=6, d_ff=2048, 
                 seq_len=512, dropout=0.1):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create the actual Transformer model
        self.model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nheads=nheads,
            nlayers=nlayers,
            d_ff=d_ff,
            seq_len=seq_len,
            dropout=dropout,
            attention_type=attention_type
        )
        
        # Temperature scheduling configuration
        self.attention_type = attention_type
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_decay_steps = temperature_decay_steps
        self.current_step = 0
        
        # Set initial temperature if using Gumbel
        if self.attention_type == AttentionType.GUMBEL:
            self.model.set_temperature(initial_temperature)

    def get_temperature_schedule(self, step):
        """Get current temperature based on schedule type"""
        if self.attention_type == AttentionType.STANDARD:
            return 1.0  # Temperature doesn't matter for standard attention
                    
        else :
            decay_ratio = step / self.temperature_decay_steps
            return self.initial_temperature * (
                (self.final_temperature / self.initial_temperature) ** decay_ratio
            )
            
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0)  # peak LR

        def lr_lambda(step):
            warmup_steps = 4000
            d_model = 512
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]
