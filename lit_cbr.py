from original_cbr import CBR_RNN
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import numpy as np


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
        
        self.use_gumbel_softmax = use_gumbel_softmax
        self.initial_temp = initial_temp if use_gumbel_softmax else None
        self.final_temp = final_temp if use_gumbel_softmax else None
        self.temp_decay = temp_decay if use_gumbel_softmax else None
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.vocab_size = vocab_size
        
        # Simple tracking
        self.step_count = 0
        self.loss_history = []
        
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
    
    def check_training_health(self, loss, output_logits, targets, stage):
        """Essential training health checks"""
        
        # 1. Loss monitoring
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"CRITICAL: Invalid loss detected: {loss}")
            return False
        
        if loss.item() > 15:
            print(f"WARNING: Very high loss: {loss.item():.4f}")
            
        # Track loss history for training dynamics
        if stage == "train":
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]
        
        # 2. Output logits health
        if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
            print(f"CRITICAL: Invalid values in model outputs")
            return False
            
        # Check if model is learning (logits should have some range)
        logit_max = output_logits.max(dim=-1)[0]
        logit_min = output_logits.min(dim=-1)[0]
        logit_range = (logit_max - logit_min).mean().item()
        
        if logit_range < 0.1:
            print(f"WARNING: Very small logit range: {logit_range:.4f} - model may not be learning")
            
        self.log(f"{stage}_logit_range", logit_range)
        
        # 3. Prediction quality
        predictions = output_logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Check vocabulary usage
        unique_preds = torch.unique(predictions).numel()
        vocab_usage = unique_preds / self.vocab_size
        
        if vocab_usage < 0.01:
            print(f"WARNING: Using only {unique_preds}/{self.vocab_size} vocabulary tokens")
            
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        self.log(f"{stage}_vocab_usage", vocab_usage)
        
        return True
    
    def check_input_data(self, sequences, targets, stage):
        """Basic input data validation"""
        
        # Check token ranges
        if sequences.min() < 0 or sequences.max() >= self.vocab_size:
            print(f"ERROR: Invalid token indices - min: {sequences.min()}, max: {sequences.max()}")
            return False
            
        if targets.min() < 0 or targets.max() >= self.vocab_size:
            print(f"ERROR: Invalid target indices - min: {targets.min()}, max: {targets.max()}")
            return False
        
        # Log basic input stats
        self.log(f"{stage}_seq_len", sequences.shape[1])
        self.log(f"{stage}_batch_size", sequences.shape[0])
        
        return True
    
    def _shared_step(self, batch, stage):
        self.step_count += 1
        
        sequences, targets = batch
        
        # Basic input validation
        if not self.check_input_data(sequences, targets, stage):
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
        
        # Compute loss
        try:
            lm_loss = self.criterion(output_flat, targets_flat)
        except Exception as e:
            print(f"ERROR in loss computation: {e}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Essential health checks
        if not self.check_training_health(lm_loss, output_flat, targets_flat, stage):
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Compute perplexity
        ppl = torch.exp(torch.clamp(lm_loss, max=10))
        
        # Log main metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=(stage == "train"), 
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        
        return lm_loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        """Simple epoch-end checks"""
        # Check loss trends
        if len(self.loss_history) >= 20:
            recent_losses = self.loss_history[-20:]
            loss_trend = recent_losses[-1] - recent_losses[0]
            loss_std = np.std(recent_losses)
            
            self.log("loss_trend", loss_trend)
            self.log("loss_stability", loss_std)
            
            if loss_std < 1e-6:
                print("WARNING: Loss appears stuck (very low variance)")
        
        # Comprehensive parameter and gradient health check
        param_norms = []
        weight_changes = []
        
        for name, param in self.named_parameters():
            param_norm = param.norm().item()
            param_norms.append(param_norm)
            
            # Check for parameter explosion
            if param_norm > 100:
                print(f"WARNING: Large parameter norm in {name}: {param_norm:.2f}")
            
            # Check for dead parameters (no variance)
            if param.numel() > 1:
                param_std = param.std().item()
                if param_std < 1e-7:
                    print(f"WARNING: Dead parameters in {name}: std={param_std:.2e}")
        
        if param_norms:
            self.log("max_param_norm", max(param_norms))
            self.log("avg_param_norm", np.mean(param_norms))
            
        print(f"Epoch {self.current_epoch}: Max param norm = {max(param_norms):.4f}, Avg = {np.mean(param_norms):.4f}")

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False
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
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Optimizer step with gradient monitoring"""
        # Compute gradients
        optimizer_closure()
        
        # Check gradients
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
        
        if total_norm > 1000:
            print(f"CRITICAL: Gradient explosion: {total_norm:.2f} - skipping step")
            optimizer.zero_grad()
            return
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
        
        # Log gradient norm
        self.log("grad_norm", total_norm)
        
        if total_norm > 10:
            print(f"WARNING: Large gradients: {total_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()