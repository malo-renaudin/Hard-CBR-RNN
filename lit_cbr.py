from original_cbr import CBR_RNN
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import numpy as np


class CBRLanguageModel(pl.LightningModule):
    """PyTorch Lightning module for training CueBasedRNNModel with extensive debugging"""
    
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
        self.criterion = nn.CrossEntropyLoss()
        
        # Debug counters and tracking
        self.debug_step = 0
        self.last_loss = None
        self.loss_history = []
        self.stuck_counter = 0
        self.vocab_size = vocab_size
        
        # Health monitoring thresholds
        self.loss_spike_threshold = 2.0  # Factor increase from previous loss
        self.grad_explosion_threshold = 10.0
        self.param_explosion_threshold = 100.0
        self.stuck_loss_threshold = 1e-6  # Minimum change to consider progress
        self.stuck_loss_patience = 50  # Steps before flagging stuck training
        
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
    
    def check_tensor_health(self, tensor, name, stage):
        """Enhanced tensor health checking with more comprehensive diagnostics"""
        batch_size = tensor.shape[0] if len(tensor.shape) > 0 else 1
        
        # Basic NaN/Inf checks
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0:
            self.log(f"{stage}_nan_{name}", nan_count / tensor.numel())
            print(f"CRITICAL: {nan_count}/{tensor.numel()} NaN values in {name} during {stage}")
            
        if inf_count > 0:
            self.log(f"{stage}_inf_{name}", inf_count / tensor.numel())
            print(f"CRITICAL: {inf_count}/{tensor.numel()} Inf values in {name} during {stage}")
            
        # Statistical analysis
        tensor_max = tensor.max().item()
        tensor_min = tensor.min().item()
        tensor_mean = tensor.mean().item()
        tensor_std = tensor.std().item()
        tensor_median = tensor.median().item()
        
        # Percentile analysis for outlier detection
        tensor_flat = tensor.flatten()
        tensor_99 = torch.quantile(tensor_flat, 0.99).item()
        tensor_01 = torch.quantile(tensor_flat, 0.01).item()
        
        # Log comprehensive statistics
        self.log(f"{stage}_{name}_max", tensor_max)
        self.log(f"{stage}_{name}_min", tensor_min)
        self.log(f"{stage}_{name}_mean", tensor_mean)
        self.log(f"{stage}_{name}_std", tensor_std)
        self.log(f"{stage}_{name}_median", tensor_median)
        self.log(f"{stage}_{name}_p99", tensor_99)
        self.log(f"{stage}_{name}_p01", tensor_01)
        self.log(f"{stage}_{name}_range", tensor_max - tensor_min)
        
        # Advanced health checks
        self._check_tensor_distribution(tensor, name, stage)
        self._check_tensor_patterns(tensor, name, stage)
        
        # Critical warnings
        if abs(tensor_max) > 100:
            print(f"WARNING: Extreme max value in {name}: {tensor_max}")
        if abs(tensor_min) > 100:
            print(f"WARNING: Extreme min value in {name}: {tensor_min}")
        if tensor_std > 20:
            print(f"WARNING: Very high variance in {name}: std={tensor_std}")
        if abs(tensor_mean) > 50:
            print(f"WARNING: Large mean in {name}: mean={tensor_mean}")
            
        # Check for degenerate distributions
        if tensor_std < 1e-6 and tensor.numel() > 1:
            print(f"WARNING: Near-zero variance in {name}: std={tensor_std}")
            self.log(f"{stage}_{name}_degenerate", 1.0)
    
    def _check_tensor_distribution(self, tensor, name, stage):
        """Check for distribution anomalies"""
        tensor_flat = tensor.flatten()
        
        # Check for dead neurons (all zeros)
        zero_count = (tensor_flat == 0).sum().item()
        zero_ratio = zero_count / tensor.numel()
        self.log(f"{stage}_{name}_zero_ratio", zero_ratio)
        
        if zero_ratio > 0.5:
            print(f"WARNING: {name} has {zero_ratio:.2%} zero values - possible dead neurons")
            
        # Check for saturated activations (if this looks like activations)
        if "output" in name.lower() or "hidden" in name.lower():
            saturated_high = (tensor_flat > 0.99).sum().item() / tensor.numel()
            saturated_low = (tensor_flat < -0.99).sum().item() / tensor.numel()
            self.log(f"{stage}_{name}_saturated_high", saturated_high)
            self.log(f"{stage}_{name}_saturated_low", saturated_low)
        
        # Check sparsity
        if tensor.numel() > 100:
            sparsity = (torch.abs(tensor_flat) < 1e-6).sum().item() / tensor.numel()
            self.log(f"{stage}_{name}_sparsity", sparsity)
    
    def _check_tensor_patterns(self, tensor, name, stage):
        """Check for suspicious patterns in tensors"""
        if len(tensor.shape) < 2:
            return
            
        # Check for repeated patterns (could indicate bugs)
        if tensor.shape[0] > 1:
            # Check if all samples in batch are identical
            first_sample = tensor[0]
            identical_samples = torch.all(tensor == first_sample.unsqueeze(0), dim=tuple(range(1, len(tensor.shape))))
            identical_ratio = identical_samples.sum().item() / tensor.shape[0]
            self.log(f"{stage}_{name}_identical_samples", identical_ratio)
            
            if identical_ratio > 0.5:
                print(f"WARNING: {name} has {identical_ratio:.2%} identical samples")
    
    def check_model_capacity_utilization(self, stage):
        """Check how well we're utilizing model capacity"""
        total_params = 0
        active_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            # Consider parameters "active" if they have reasonable gradients
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:  # Threshold for "active"
                    active_params += param.numel()
        
        utilization = active_params / total_params if total_params > 0 else 0
        self.log(f"{stage}_capacity_utilization", utilization)
        
        if utilization < 0.1:
            print(f"WARNING: Low capacity utilization: {utilization:.2%}")
    
    def check_loss_dynamics(self, current_loss, stage):
        """Monitor loss dynamics for anomalies"""
        if stage != "train":
            return
            
        self.loss_history.append(current_loss.item())
        
        # Keep only recent history
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        if len(self.loss_history) < 2:
            return
        
        # Check for loss spikes
        if self.last_loss is not None:
            loss_change = current_loss.item() / (self.last_loss + 1e-8)
            self.log(f"{stage}_loss_change_ratio", loss_change)
            
            if loss_change > self.loss_spike_threshold:
                print(f"WARNING: Loss spike detected: {self.last_loss:.4f} -> {current_loss.item():.4f}")
                self.log(f"{stage}_loss_spike", 1.0)
        
        # Check if loss is stuck
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_variance = np.var(recent_losses)
            
            if loss_variance < self.stuck_loss_threshold:
                self.stuck_counter += 1
                if self.stuck_counter > self.stuck_loss_patience:
                    print(f"WARNING: Loss appears stuck (variance: {loss_variance:.8f})")
                    self.log(f"{stage}_loss_stuck", 1.0)
            else:
                self.stuck_counter = 0
        
        # Track loss smoothness
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            loss_smoothness = np.std(recent_losses) / (np.mean(recent_losses) + 1e-8)
            self.log(f"{stage}_loss_smoothness", loss_smoothness)
        
        self.last_loss = current_loss.item()
    
    def check_vocabulary_coverage(self, predictions, targets, stage):
        """Monitor vocabulary usage patterns"""
        unique_preds = torch.unique(predictions).numel()
        unique_targets = torch.unique(targets).numel()
        
        vocab_coverage_pred = unique_preds / self.vocab_size
        vocab_coverage_target = unique_targets / self.vocab_size
        
        self.log(f"{stage}_vocab_coverage_predictions", vocab_coverage_pred)
        self.log(f"{stage}_vocab_coverage_targets", vocab_coverage_target)
        
        # Check for degenerate predictions (predicting only few tokens)
        if vocab_coverage_pred < 0.01:  # Less than 1% of vocabulary
            print(f"WARNING: Model using only {unique_preds} out of {self.vocab_size} vocabulary tokens")
            self.log(f"{stage}_degenerate_predictions", 1.0)
        
        # Check prediction distribution
        pred_counts = torch.bincount(predictions.flatten(), minlength=self.vocab_size)
        pred_entropy = -torch.sum(pred_counts / pred_counts.sum() * torch.log(pred_counts / pred_counts.sum() + 1e-8))
        max_entropy = math.log(self.vocab_size)  # Uniform distribution entropy
        normalized_entropy = pred_entropy / max_entropy
        
        self.log(f"{stage}_prediction_entropy_normalized", normalized_entropy)
        
    def check_learning_progress(self, stage):
        """Monitor if the model is actually learning"""
        if stage != "train" or self.debug_step < 100:
            return
        
        # Check gradient-to-parameter ratio (should be reasonable)
        total_grad_norm = 0
        total_param_norm = 0
        
        for param in self.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                total_param_norm += param.norm().item() ** 2
        
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5
        
        if total_param_norm > 0:
            grad_param_ratio = total_grad_norm / total_param_norm
            self.log(f"{stage}_grad_param_ratio", grad_param_ratio)
            
            # Healthy range is typically 1e-6 to 1e-2
            if grad_param_ratio < 1e-8:
                print(f"WARNING: Very small gradients relative to parameters: {grad_param_ratio:.2e}")
            elif grad_param_ratio > 1e-1:
                print(f"WARNING: Very large gradients relative to parameters: {grad_param_ratio:.2e}")
    
    def _shared_step(self, batch, stage):
        self.debug_step += 1
        
        sequences, targets = batch
        
        # Enhanced input validation
        self.check_tensor_health(sequences.float(), "input_sequences", stage)
        self.check_tensor_health(targets.float(), "input_targets", stage)
        
        # Check for valid token ranges
        if sequences.min() < 0 or sequences.max() >= self.vocab_size:
            print(f"ERROR: Token indices out of range: min={sequences.min()}, max={sequences.max()}")
            self.log(f"{stage}_invalid_tokens", 1.0)
        
        # Log detailed input statistics
        self.log(f"{stage}_batch_size", sequences.shape[0])
        self.log(f"{stage}_seq_length", sequences.shape[1])
        self.log(f"{stage}_seq_max_token", sequences.max().item())
        self.log(f"{stage}_seq_min_token", sequences.min().item())
        self.log(f"{stage}_seq_unique_tokens", torch.unique(sequences).numel())
        
        # Check for data loader issues
        if torch.all(sequences == 0):
            print("WARNING: All input tokens are padding/zero")
            self.log(f"{stage}_all_padding", 1.0)
        
        sequences = sequences.transpose(0, 1)
        targets = targets.transpose(0, 1)
        
        initial_cache = self.model.init_cache(sequences)
        
        # Enhanced cache health checking
        for i, cache_tensor in enumerate(initial_cache):
            self.check_tensor_health(cache_tensor, f"initial_cache_{i}", stage)
        
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
            
        # Forward pass with error handling
        try:
            output, final_hidden = self.model.forward(**forward_kwargs)
        except Exception as e:
            print(f"CRITICAL ERROR in forward pass: {e}")
            print(f"Input shapes: sequences={sequences.shape}, cache shapes={[c.shape for c in initial_cache]}")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Comprehensive output checking
        self.check_tensor_health(output, "model_output", stage)
        self.check_tensor_health(final_hidden, "final_hidden", stage)
        
        # Check output logits
        output_flat = output.reshape(-1, output.size(-1))
        targets_flat = targets.reshape(-1)
        
        self.check_tensor_health(output_flat, "output_logits", stage)
        
        # Enhanced prediction analysis
        output_probs = F.softmax(output_flat, dim=-1)
        max_probs, predictions = output_probs.max(dim=-1)
        
        self.log(f"{stage}_max_prob_mean", max_probs.mean().item())
        self.log(f"{stage}_max_prob_std", max_probs.std().item())
        self.log(f"{stage}_max_prob_min", max_probs.min().item())
        
        # Check for overconfident predictions
        overconfident_ratio = (max_probs > 0.99).float().mean()
        self.log(f"{stage}_overconfident_predictions", overconfident_ratio)
        
        # Enhanced entropy analysis
        entropy = -(output_probs * torch.log(output_probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        self.log(f"{stage}_prediction_entropy", entropy.item())
        self.log(f"{stage}_prediction_entropy_normalized", normalized_entropy.item())
        
        # Vocabulary coverage analysis
        self.check_vocabulary_coverage(predictions, targets_flat, stage)
        
        # Compute loss with enhanced error handling
        try:
            lm_loss = self.criterion(output_flat, targets_flat)
        except Exception as e:
            print(f"CRITICAL ERROR in loss computation: {e}")
            print(f"Output shape: {output_flat.shape}, Targets shape: {targets_flat.shape}")
            print(f"Output range: [{output_flat.min():.4f}, {output_flat.max():.4f}]")
            print(f"Targets range: [{targets_flat.min()}, {targets_flat.max()}]")
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
        
        # Enhanced loss validation
        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            print(f"CRITICAL: Invalid loss detected: {lm_loss.item()}")
            self.log(f"{stage}_invalid_loss", 1.0)
            return torch.tensor(float('inf'), device=sequences.device, requires_grad=True)
            
        # Check loss dynamics
        self.check_loss_dynamics(lm_loss, stage)
        
        if lm_loss.item() > 15:  # Very high loss for language modeling
            print(f"WARNING: Extremely high loss: {lm_loss.item():.4f}")
            self.log(f"{stage}_extreme_loss", 1.0)
        
        if lm_loss.item() < 0.01:  # Suspiciously low loss
            print(f"WARNING: Suspiciously low loss: {lm_loss.item():.4f}")
            self.log(f"{stage}_suspicious_low_loss", 1.0)
        
        # Safe perplexity computation
        try:
            clamped_loss = torch.clamp(lm_loss, max=10)
            ppl = torch.exp(clamped_loss)
        except:
            ppl = torch.tensor(float('inf'))
            
        # Enhanced accuracy metrics
        predictions = output_flat.argmax(dim=-1)
        accuracy = (predictions == targets_flat).float().mean()
        
        # Top-k accuracies
        _, top5_preds = output_flat.topk(5, dim=-1)
        top5_accuracy = (top5_preds == targets_flat.unsqueeze(-1)).any(dim=-1).float().mean()
        
        # Log all metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=(stage == "train"), 
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        self.log(f"{stage}_top5_accuracy", top5_accuracy)
        
        # Monitor learning progress
        self.check_learning_progress(stage)
        
        # Monitor model capacity utilization
        if self.debug_step % 100 == 0:
            self.check_model_capacity_utilization(stage)
        
        # Enhanced gradient monitoring
        if stage == "train" and self.debug_step % 50 == 0:
            self._monitor_gradients(stage)
        
        return lm_loss
    
    def _monitor_gradients(self, stage):
        """Comprehensive gradient monitoring"""
        total_params = 0
        grad_norm_sq = 0
        zero_grad_params = 0
        large_grad_params = 0
        
        layer_grad_norms = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                grad_norm_sq += param_grad_norm ** 2
                total_params += param.numel()
                
                # Count problematic gradients
                if param_grad_norm < 1e-8:
                    zero_grad_params += param.numel()
                elif param_grad_norm > 10:
                    large_grad_params += param.numel()
                
                # Group by layer type
                layer_type = name.split('.')[0] if '.' in name else name
                if layer_type not in layer_grad_norms:
                    layer_grad_norms[layer_type] = []
                layer_grad_norms[layer_type].append(param_grad_norm)
                
                # Log individual parameter gradients less frequently
                if self.debug_step % 200 == 0:
                    self.log(f"grad_norm_{name.replace('.', '_')}", param_grad_norm)
        
        total_grad_norm = grad_norm_sq ** 0.5
        
        # Log gradient statistics
        self.log(f"{stage}_total_grad_norm", total_grad_norm)
        self.log(f"{stage}_zero_grad_ratio", zero_grad_params / max(total_params, 1))
        self.log(f"{stage}_large_grad_ratio", large_grad_params / max(total_params, 1))
        
        # Log layer-wise gradient norms
        for layer_type, grad_norms in layer_grad_norms.items():
            avg_grad_norm = np.mean(grad_norms)
            max_grad_norm = np.max(grad_norms)
            self.log(f"{stage}_grad_norm_{layer_type}_avg", avg_grad_norm)
            self.log(f"{stage}_grad_norm_{layer_type}_max", max_grad_norm)
        
        # Check for gradient explosion
        if total_grad_norm > self.grad_explosion_threshold:
            print(f"WARNING: Large gradient norm detected: {total_grad_norm:.4f}")
            self.log(f"{stage}_grad_explosion", 1.0)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        """Enhanced epoch-end diagnostics"""
        print(f"\n=== EPOCH {self.current_epoch} HEALTH REPORT ===")
        
        # Check model parameter health
        param_stats = {}
        for name, param in self.named_parameters():
            if param.data is not None:
                param_norm = param.data.norm(2).item()
                param_max = param.data.abs().max().item()
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                
                param_stats[name] = {
                    'norm': param_norm,
                    'max': param_max,
                    'mean': param_mean,
                    'std': param_std
                }
                
                self.log(f"param_norm_{name.replace('.', '_')}", param_norm)
                self.log(f"param_max_{name.replace('.', '_')}", param_max)
                
                # Check for parameter explosion
                if param_norm > self.param_explosion_threshold:
                    print(f"WARNING: Large parameter norm in {name}: {param_norm:.4f}")
                    self.log(f"param_explosion_{name.replace('.', '_')}", 1.0)
                
                # Check for dead parameters
                if param_std < 1e-6:
                    print(f"WARNING: Near-zero parameter variance in {name}: {param_std:.2e}")
                    self.log(f"param_dead_{name.replace('.', '_')}", 1.0)
        
        # Log current learning rate
        if hasattr(self, 'optimizers'):
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log("current_lr", current_lr)
            print(f"Current Learning Rate: {current_lr:.2e}")
        
        # Memory usage tracking
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            self.log("gpu_memory_allocated_gb", memory_allocated)
            self.log("gpu_memory_cached_gb", memory_cached)
            print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        print("=" * 50)
        
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
        """Enhanced optimizer step with comprehensive gradient monitoring"""
        # Compute gradients
        optimizer_closure()
        
        # Pre-clipping gradient analysis
        total_norm_before = 0
        param_count = 0
        max_grad_norm = 0
        min_grad_norm = float('inf')
        
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm_before += param_norm ** 2
                param_count += 1
                max_grad_norm = max(max_grad_norm, param_norm)
                min_grad_norm = min(min_grad_norm, param_norm) if min_grad_norm != float('inf') else param_norm
        
        total_norm_before = total_norm_before ** 0.5
        
        self.log("grad_norm_before_clip", total_norm_before)
        self.log("grad_norm_max_param", max_grad_norm)
        self.log("grad_norm_min_param", min_grad_norm if min_grad_norm != float('inf') else 0)
        self.log("grad_norm_avg_param", total_norm_before / max(param_count, 1))
        
        # Apply gradient clipping with monitoring
        if total_norm_before > 1000:
            print(f"CRITICAL: Severe gradient explosion: {total_norm_before:.2f}")
            print("SKIPPING OPTIMIZER STEP to prevent model corruption")
            optimizer.zero_grad()
            self.log("optimizer_steps_skipped", 1.0)
            return
        
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
        self.log("grad_norm_after_clip", clipped_norm.item())
        
        # Calculate clipping ratio
        clipping_ratio = clipped_norm / max(total_norm_before, 1e-8)
        self.log("grad_clipping_ratio", clipping_ratio)
        
        if clipping_ratio < 0.5:
            print(f"INFO: Significant gradient clipping: {total_norm_before:.4f} -> {clipped_norm:.4f}")
        
        # Check gradient health after clipping
        if clipped_norm > 10:
            print(f"WARNING: Still large gradients after clipping: {clipped_norm:.4f}")
            self.log("grad_still_large_after_clip", 1.0)
        
        if clipped_norm < 1e-8:
            print(f"WARNING: Extremely small gradients: {clipped_norm:.2e}")
            self.log("grad_extremely_small", 1.0)
        
        # Perform optimizer step
        try:
            optimizer.step()
            self.log("optimizer_step_success", 1.0)
        except Exception as e:
            print(f"ERROR in optimizer step: {e}")
            self.log("optimizer_step_error", 1.0)
            
        optimizer.zero_grad()
        
        # Post-step parameter health check (less frequent)
        if batch_idx % 100 == 0:
            self._post_step_param_check()
    
    def _post_step_param_check(self):
        """Check parameter health after optimizer step"""
        total_param_norm = 0
        nan_params = 0
        inf_params = 0
        
        for name, param in self.named_parameters():
            param_norm = param.data.norm(2).item()
            total_param_norm += param_norm ** 2
            
            if torch.isnan(param).any():
                nan_params += 1
                print(f"CRITICAL: NaN in parameters {name} after optimizer step")
                
            if torch.isinf(param).any():
                inf_params += 1
                print(f"CRITICAL: Inf in parameters {name} after optimizer step")
        
        total_param_norm = total_param_norm ** 0.5
        
        self.log("post_step_param_norm", total_param_norm)
        self.log("post_step_nan_params", nan_params)
        self.log("post_step_inf_params", inf_params)
        
        if nan_params > 0 or inf_params > 0:
            print(f"CRITICAL: Model corruption detected! NaN params: {nan_params}, Inf params: {inf_params}")
            self.log("model_corruption_detected", 1.0)
    
    def on_train_start(self):
        """Initial model health check before training starts"""
        print("\n=== PRE-TRAINING MODEL HEALTH CHECK ===")
        
        # Check initial parameter distribution
        total_params = 0
        param_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            param_stats['min'].append(param.min().item())
            param_stats['max'].append(param.max().item())
            param_stats['mean'].append(param.mean().item())
            param_stats['std'].append(param.std().item())
            
            # Log initial parameter statistics
            self.log(f"init_param_{name.replace('.', '_')}_norm", param.norm().item())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Parameter ranges: [{min(param_stats['min']):.4f}, {max(param_stats['max']):.4f}]")
        print(f"Average parameter std: {np.mean(param_stats['std']):.4f}")
        
        # Check model architecture
        self.log("model_total_params", float(total_params))
        self.log("model_layers", float(self.hparams.nlayers))
        self.log("model_hidden_size", float(self.hparams.nhid))
        self.log("vocab_size", float(self.vocab_size))
        
        print("=" * 45)
    
    def on_validation_epoch_end(self):
        """Validation epoch end checks"""
        if hasattr(self.trainer, 'callback_metrics'):
            val_loss = self.trainer.callback_metrics.get('val_loss')
            train_loss = self.trainer.callback_metrics.get('train_loss')
            
            if val_loss is not None and train_loss is not None:
                # Check for overfitting
                loss_gap = val_loss - train_loss
                self.log("train_val_loss_gap", loss_gap)
                
                if loss_gap > 2.0:
                    print(f"WARNING: Large train/val loss gap: {loss_gap:.4f} - possible overfitting")
                    self.log("potential_overfitting", 1.0)
                
                # Check for underfitting
                if val_loss > 10 and self.current_epoch > 5:
                    print(f"WARNING: High validation loss after epoch {self.current_epoch}: {val_loss:.4f}")
                    self.log("potential_underfitting", 1.0)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Per-batch monitoring for critical issues"""
        if batch_idx % 500 == 0:  # Every 500 batches
            # Quick memory check
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_used > 0.95:
                    print(f"WARNING: Very high GPU memory usage: {memory_used:.1%}")
                    self.log("high_memory_usage", memory_used)
            
            # Check if loss is exploding
            if outputs is not None and hasattr(outputs, 'item'):
                current_loss = outputs.item()
                if current_loss > 50:
                    print(f"CRITICAL: Loss explosion detected: {current_loss:.2f}")
                    self.log("loss_explosion_critical", current_loss)
    
    def state_dict(self):
        """Enhanced state dict with health metadata"""
        state = super().state_dict()
        
        # Add training health metadata
        health_metadata = {
            'training_step': self.debug_step,
            'last_loss': self.last_loss,
            'loss_history_recent': self.loss_history[-10:] if self.loss_history else [],
            'stuck_counter': self.stuck_counter,
            'vocab_size': self.vocab_size,
            'training_epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0
        }
        
        state['health_metadata'] = health_metadata
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Enhanced state dict loading with health restoration"""
        # Extract health metadata if present
        health_metadata = state_dict.pop('health_metadata', {})
        
        # Load model state
        result = super().load_state_dict(state_dict, strict)
        
        # Restore health monitoring state
        if health_metadata:
            self.debug_step = health_metadata.get('training_step', 0)
            self.last_loss = health_metadata.get('last_loss', None)
            self.loss_history = health_metadata.get('loss_history_recent', [])
            self.stuck_counter = health_metadata.get('stuck_counter', 0)
            
            print(f"Restored training health state: step {self.debug_step}, last_loss {self.last_loss}")
        
        return result
    
    def on_exception(self, trainer, pl_module, exception):
        """Handle training exceptions with detailed diagnostics"""
        print(f"\n=== TRAINING EXCEPTION DIAGNOSTICS ===")
        print(f"Exception: {type(exception).__name__}: {exception}")
        print(f"Training step: {self.debug_step}")
        print(f"Current epoch: {self.current_epoch}")
        
        # Log current model state
        param_norms = []
        grad_norms = []
        
        for name, param in self.named_parameters():
            if param is not None:
                param_norm = param.norm().item()
                param_norms.append(param_norm)
                
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                # Check for NaN/Inf in parameters
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                if torch.isinf(param).any():
                    print(f"Inf detected in parameter: {name}")
        
        if param_norms:
            print(f"Parameter norms - Max: {max(param_norms):.4f}, Min: {min(param_norms):.4f}")
        if grad_norms:
            print(f"Gradient norms - Max: {max(grad_norms):.4f}, Min: {min(grad_norms):.4f}")
        
        print("=" * 45)
        
        # Log exception for tracking
        self.log("training_exception", 1.0)
    
    def manual_backward(self, loss, *args, **kwargs):
        """Override manual backward with additional safety checks"""
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"CRITICAL: Attempting to backward through invalid loss: {loss.item()}")
            self.log("backward_invalid_loss", 1.0)
            return
        
        if loss.item() > 100:
            print(f"WARNING: Backward through very high loss: {loss.item():.4f}")
            self.log("backward_high_loss", loss.item())
        
        try:
            super().manual_backward(loss, *args, **kwargs)
        except Exception as e:
            print(f"ERROR in manual_backward: {e}")
            self.log("backward_error", 1.0)
            raise
    
    def get_training_diagnostics(self):
        """Return comprehensive training diagnostics"""
        diagnostics = {
            'training_step': self.debug_step,
            'current_epoch': getattr(self, 'current_epoch', 0),
            'last_loss': self.last_loss,
            'loss_history': self.loss_history[-20:] if self.loss_history else [],
            'stuck_counter': self.stuck_counter,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # Add parameter health summary
        param_health = {}
        for name, param in self.named_parameters():
            if param is not None:
                param_health[name] = {
                    'norm': param.norm().item(),
                    'max': param.abs().max().item(),
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'has_nan': torch.isnan(param).any().item(),
                    'has_inf': torch.isinf(param).any().item(),
                }
        
        diagnostics['parameter_health'] = param_health
        
        # Add gradient health if available
        grad_health = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_health[name] = {
                    'norm': param.grad.norm().item(),
                    'max': param.grad.abs().max().item(),
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'has_nan': torch.isnan(param.grad).any().item(),
                    'has_inf': torch.isinf(param.grad).any().item(),
                }
        
        if grad_health:
            diagnostics['gradient_health'] = grad_health
            
        return diagnostics
    
    def print_model_summary(self):
        """Print detailed model architecture and health summary"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE & HEALTH SUMMARY")
        print("="*60)
        
        # Architecture info
        print(f"Vocabulary Size: {self.vocab_size:,}")
        print(f"Input Embedding Dim: {self.hparams.ninp}")
        print(f"Hidden Size: {self.hparams.nhid}")
        print(f"Number of Layers: {self.hparams.nlayers}")
        print(f"Dropout: {self.hparams.dropout}")
        if hasattr(self.hparams, 'nheads'):
            print(f"Number of Heads: {self.hparams.nheads}")
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        
        # Memory estimate
        param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32
        print(f"Estimated Parameter Memory: {param_memory:.2f} GB")
        
        # Training state
        print(f"\nTraining State:")
        print(f"Current Step: {self.debug_step}")
        print(f"Current Epoch: {getattr(self, 'current_epoch', 0)}")
        print(f"Last Loss: {self.last_loss if self.last_loss else 'N/A'}")
        print(f"Learning Rate: {self.lr}")
        
        # Health indicators
        print(f"\nHealth Indicators:")
        print(f"Loss History Length: {len(self.loss_history)}")
        print(f"Stuck Counter: {self.stuck_counter}")
        print(f"Gumbel Softmax: {self.use_gumbel_softmax}")
        
        print("="*60)