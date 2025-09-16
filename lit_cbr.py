from original_cbr import CBR_RNN
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
            verbose=True
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

    
    