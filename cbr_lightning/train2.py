#!/usr/bin/env python3
"""
Universal Training Script for CBR_RNN, Transformer, and LSTM models
with PyTorch Lightning and simple file-based logging
"""

import os
import argparse
import yaml
import random
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import datasets
import pickle

# Import your models
from model_lightning import CBR_RNN, Transformer, LSTM


class SimpleFileLogger:
    """Simple file-based logger to replace MLflow/TensorBoard"""
    
    def __init__(self, experiment_name: str, log_dir: str = "./training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create experiment directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.hyperparams_file = self.experiment_dir / "hyperparameters.json"
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.experiment_info_file = self.experiment_dir / "experiment_info.txt"
        
        print(f"Logging to: {self.experiment_dir}")
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters to JSON file"""
        with open(self.hyperparams_file, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        print(f"Hyperparameters logged to: {self.hyperparams_file}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to JSONL file"""
        log_entry = {
            'step': step,
            'timestamp': datetime.datetime.now().isoformat(),
            **metrics
        }
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')
    
    def log_experiment_info(self, info: str):
        """Log general experiment information"""
        with open(self.experiment_info_file, 'a') as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {info}\n")


class SimpleMetricsCallback(pl.Callback):
    """Callback to log metrics to file"""
    
    def __init__(self, logger: SimpleFileLogger):
        self.logger = logger
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at epoch end"""
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value
        
        if hasattr(pl_module, 'temperature'):
            metrics['temperature'] = pl_module.temperature
            
        self.logger.log_metrics(metrics, trainer.current_epoch)


class UniversalDataModule(pl.LightningDataModule):
    """Data module that handles your tokenized dataset"""
    
    def __init__(self, dataset_path: str, tokenizer_path: str, batch_size: int = 32, 
                 num_workers: int = 4, max_length: Optional[int] = None):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
    def setup(self, stage: Optional[str] = None):
        # Load dataset
        self.dataset = datasets.load_from_disk(self.dataset_path)
        
        # Load tokenizer
        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        
        self.vocab_size = len(self.tokenizer['word2idx'])
        
    def collate_fn(self, batch):
        """Custom collate function to handle your data format"""
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]
        
        # Convert to tensors and stack
        input_tensor = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in input_ids])
        target_tensor = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in target_ids])
        
        # Truncate if max_length specified
        if self.max_length:
            input_tensor = input_tensor[:, :self.max_length]
            target_tensor = target_tensor[:, :self.max_length]
        
        return input_tensor, target_tensor
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )


class ModelFactory:
    """Factory to create models based on configuration"""
    
    @staticmethod
    def create_model(model_type: str, model_config: Dict[str, Any], vocab_size: int) -> pl.LightningModule:
        """Create model instance based on type and configuration"""
        
        model_config = model_config.copy()
        model_config['ntoken'] = vocab_size  # Add vocab size to config
        
        if model_type.upper() == 'CBR_RNN':
            return CBR_RNN(**model_config)
            
        elif model_type.upper() == 'TRANSFORMER':
            # Map config keys for Transformer
            transformer_config = {
                'vocab_size': vocab_size,
                'd_model': model_config.get('ninp', 256),
                'n_heads': model_config.get('nheads', 8),
                'n_layers': model_config.get('n_layers', 6),
                'd_ff': model_config.get('d_ff', 1024),
                'max_seq_len': model_config.get('seq_len', 128),
                'dropout': model_config.get('dropout', 0.1),
                'learning_rate': model_config.get('learning_rate', 1e-3),
                'temperature': model_config.get('temperature', 1.0),
                'gumbel_softmax': model_config.get('gumbel_softmax', False)
            }
            return Transformer(**transformer_config)
            
        elif model_type.upper() == 'LSTM':
            # Map config keys for LSTM
            lstm_config = {
                'vocab_size': vocab_size,
                'embedding_dim': model_config.get('ninp', 256),
                'hidden_dim': model_config.get('nhid', 512),
                'learning_rate': model_config.get('learning_rate', 1e-3)
            }
            return LSTM(**lstm_config)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class TemperatureSchedulerCallback(pl.Callback):
    """Callback to update temperature scheduler every epoch"""
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update temperature at the start of each training epoch"""
        if hasattr(pl_module, 'temp_scheduler') and hasattr(pl_module, 'gumbel_softmax'):
            if pl_module.gumbel_softmax:
                pl_module.temp_scheduler.step()
                current_temp = pl_module.temp_scheduler.get_temperature()
                pl_module.temperature = current_temp
                
                # Log temperature to console
                print(f"Epoch {trainer.current_epoch}: Temperature = {current_temp:.6f}")


class ComprehensiveCheckpointCallback(pl.Callback):
    """Custom callback to save comprehensive checkpoints with all necessary state"""
    
    def __init__(self, dirpath: str = "./checkpoints", save_every_epoch: bool = True):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.save_every_epoch = save_every_epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save comprehensive checkpoint at the end of each epoch"""
        if not self.save_every_epoch and trainer.current_epoch % 5 != 0:
            return  # Only save every 5 epochs if save_every_epoch is False
        
        epoch = trainer.current_epoch
        filename = f"comprehensive_epoch_{epoch:03d}.ckpt"
        filepath = self.dirpath / filename
        
        # Gather all state information
        checkpoint_data = {
            # Model state
            'state_dict': pl_module.state_dict(),
            'model_config': pl_module.hparams,
            'model_type': type(pl_module).__name__,
            
            # Training state
            'epoch': epoch,
            'global_step': trainer.global_step,
            'lr_schedulers': trainer.lr_scheduler_configs,
            'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
            
            # Random states for reproducibility
            'pytorch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': torch.random.get_rng_state(),
            'python_rng_state': random.getstate(),
            
            # Model-specific states
            'current_temperature': getattr(pl_module, 'temperature', None),
            'epoch_cache': getattr(pl_module, 'epoch_cache', None),
            
            # Temperature scheduler state (if exists)
            'temp_scheduler_state': None,
            
            # Training metrics
            'train_loss': trainer.callback_metrics.get('train_loss', None),
            'val_loss': trainer.callback_metrics.get('val_loss', None),
            'train_ppl': trainer.callback_metrics.get('train_ppl', None),
            'val_ppl': trainer.callback_metrics.get('val_ppl', None),
        }
        
        # Save temperature scheduler state if it exists
        if hasattr(pl_module, 'temp_scheduler'):
            temp_scheduler = pl_module.temp_scheduler
            checkpoint_data['temp_scheduler_state'] = {
                'current_epoch': temp_scheduler.current_epoch,
                'initial_temp': temp_scheduler.initial_temp,
                'decay_rate': temp_scheduler.decay_rate,
                'final_temp': temp_scheduler.final_temp,
                'current_temp': temp_scheduler.get_temperature()
            }
        
        # CUDA random state if using GPU
        if torch.cuda.is_available():
            checkpoint_data['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        
        # Save the comprehensive checkpoint
        torch.save(checkpoint_data, filepath)
        
        print(f"Comprehensive checkpoint saved: {filepath}")


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create PyTorch Lightning callbacks"""
    callbacks = []
    
    # Comprehensive checkpoint callback (every epoch)
    checkpoint_config = config.get('checkpoint', {})
    comprehensive_checkpoint = ComprehensiveCheckpointCallback(
        dirpath=checkpoint_config.get('dirpath', './checkpoints'),
        save_every_epoch=checkpoint_config.get('save_every_epoch', True)
    )
    callbacks.append(comprehensive_checkpoint)
    
    # Standard model checkpoint (best models only)
    best_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_config.get('dirpath', './checkpoints'),
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=False,  # We handle this with comprehensive checkpoints
        verbose=False
    )
    callbacks.append(best_model_checkpoint)
    
    # Early stopping
    early_stop_config = config.get('early_stopping', {})
    if early_stop_config.get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val_loss'),
            mode=early_stop_config.get('mode', 'min'),
            patience=early_stop_config.get('patience', 10),
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def load_comprehensive_checkpoint(checkpoint_path: str, model: pl.LightningModule = None):
    """
    Load a comprehensive checkpoint with all training state
    
    Args:
        checkpoint_path: Path to the comprehensive checkpoint file
        model: Optional model instance to load state into
        
    Returns:
        Dictionary containing all checkpoint data
    """
    print(f"Loading comprehensive checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint info
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Global step: {checkpoint['global_step']}")
    print(f"  Model type: {checkpoint['model_type']}")
    
    if checkpoint.get('current_temperature') is not None:
        print(f"  Current temperature: {checkpoint['current_temperature']:.6f}")
    
    if checkpoint.get('train_loss') is not None:
        print(f"  Train loss: {checkpoint['train_loss']:.6f}")
    if checkpoint.get('val_loss') is not None:
        print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    
    # Restore random states for reproducibility
    restore_random_states(checkpoint)
    
    # Load model state if model provided
    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
        
        # Restore model-specific state
        if checkpoint.get('current_temperature') is not None:
            model.temperature = checkpoint['current_temperature']
        
        if checkpoint.get('epoch_cache') is not None:
            model.epoch_cache = checkpoint['epoch_cache']
        
        # Restore temperature scheduler state
        if checkpoint.get('temp_scheduler_state') and hasattr(model, 'temp_scheduler'):
            temp_state = checkpoint['temp_scheduler_state']
            model.temp_scheduler.current_epoch = temp_state['current_epoch']
            model.temp_scheduler.initial_temp = temp_state['initial_temp']
            model.temp_scheduler.decay_rate = temp_state['decay_rate']
            model.temp_scheduler.final_temp = temp_state['final_temp']
            print(f"  Temperature scheduler restored (epoch {temp_state['current_epoch']})")
        
        print("Model state loaded successfully")
    
    return checkpoint


def restore_random_states(checkpoint: Dict[str, Any]):
    """Restore all random number generator states for reproducibility"""
    
    # PyTorch RNG states
    if 'pytorch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['pytorch_rng_state'])
    
    if 'numpy_rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['numpy_rng_state'])
    
    # Python RNG state
    if 'python_rng_state' in checkpoint:
        import random
        random.setstate(checkpoint['python_rng_state'])
    
    # CUDA RNG states
    if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    
    print("Random states restored for reproducibility")


def resume_training_from_comprehensive_checkpoint(checkpoint_path: str, config_path: str = None):
    """
    Resume training from a comprehensive checkpoint
    
    Args:
        checkpoint_path: Path to comprehensive checkpoint
        config_path: Optional path to config file (uses checkpoint config if not provided)
    """
    
    # Load checkpoint
    checkpoint = load_comprehensive_checkpoint(checkpoint_path)
    
    # Use config from checkpoint or load from file
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Using config file: {config_path}")
    else:
        # Reconstruct config from checkpoint
        model_config = checkpoint['model_config']
        config = {
            'model': {
                'type': checkpoint['model_type'].replace('CBR_RNN', 'CBR_RNN').replace('Transformer', 'Transformer').replace('LSTM', 'LSTM'),
                'config': model_config
            },
            # Add other necessary config sections...
            'trainer': {
                'max_epochs': 100,  # Continue for more epochs
                'accelerator': 'auto',
                'devices': 'auto',
            }
        }
        print("Using config reconstructed from checkpoint")
    
    # Set up data module (you'll need dataset info)
    # This assumes your dataset paths are still valid
    data_config = config.get('data', {
        'dataset_path': 'cbr_lightning/wikitext-103-tokenized',
        'tokenizer_path': './tokenizer.pkl',
        'batch_size': 32,
        'num_workers': 4
    })
    
    data_module = UniversalDataModule(**data_config)
    data_module.setup()
    
    # Create model with same architecture
    model_type = config['model']['type']
    model_config = config['model']['config']
    model = ModelFactory.create_model(model_type, model_config, data_module.vocab_size)
    
    # Load checkpoint into model
    load_comprehensive_checkpoint(checkpoint_path, model)
    
    # Create trainer with no external logging
    trainer = pl.Trainer(
        logger=False,  # Disable built-in logger
        callbacks=create_callbacks(config),
        **config.get('trainer', {})
    )
    
    # Resume training
    print("Resuming training...")
    trainer.fit(model, datamodule=data_module)
    
    return model, trainer


def train_model(config_path: str, resume_from_checkpoint: Optional[str] = None):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from: {config_path}")
    
    # Set random seed
    if 'seed' in config:
        pl.seed_everything(config['seed'])
    
    # Create data module
    data_config = config['data']
    data_module = UniversalDataModule(**data_config)
    data_module.setup()
    
    print(f"Dataset loaded. Vocab size: {data_module.vocab_size}")
    
    # Create model
    model_type = config['model']['type']
    model_config = config['model']['config']
    model = ModelFactory.create_model(model_type, model_config, data_module.vocab_size)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {model_type}")
    print(f"Total parameters: {total_params:,}")
    
    # Print temperature scheduler info if using Gumbel softmax
    if hasattr(model, 'gumbel_softmax') and model.gumbel_softmax:
        if hasattr(model, 'temp_scheduler'):
            print(f"Using temperature scheduler:")
            print(f"  Initial temperature: {model.temp_scheduler.initial_temp}")
            print(f"  Decay rate: {model.temp_scheduler.decay_rate}")
            print(f"  Final temperature: {model.temp_scheduler.final_temp}")
        else:
            print("Warning: Gumbel softmax enabled but no temperature scheduler found!")
    
    # Create simple file logger
    logger_config = config.get('logging', {})
    experiment_name = logger_config.get('experiment_name', f'language_modeling_{model_type.lower()}')
    logger = SimpleFileLogger(experiment_name)
    
    # Log experiment info
    logger.log_experiment_info(f"Starting training for {model_type}")
    logger.log_experiment_info(f"Total parameters: {total_params:,}")
    logger.log_experiment_info(f"Vocab size: {data_module.vocab_size}")
    
    # Log hyperparameters
    hyperparams = {
        'model_type': model_type,
        'total_params': total_params,
        'vocab_size': data_module.vocab_size,
        **model_config,
        **data_config
    }
    
    # Add temperature scheduler params if applicable
    if hasattr(model, 'temp_scheduler'):
        hyperparams.update({
            'temp_initial': model.temp_scheduler.initial_temp,
            'temp_decay_rate': model.temp_scheduler.decay_rate,
            'temp_final': model.temp_scheduler.final_temp
        })
        logger.log_experiment_info(f"Using temperature scheduler: {model.temp_scheduler.initial_temp} -> {model.temp_scheduler.final_temp}")
    
    logger.log_hyperparams(hyperparams)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Add temperature scheduler callback if needed
    if hasattr(model, 'temp_scheduler'):
        callbacks.append(TemperatureSchedulerCallback())
    
    # Add simple metrics callback
    callbacks.append(SimpleMetricsCallback(logger))
    
    # Create trainer (no external logger)
    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        logger=False,  # Disable built-in logger
        callbacks=callbacks,
        **trainer_config
    )
    
    print("Starting training...")
    
    # Train model
    trainer.fit(
        model, 
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint
    )
    
    # Test model
    if config.get('run_test', True):
        print("Running test...")
        trainer.test(model, datamodule=data_module)
    
    print("Training completed!")
    
    # Save final model
    save_path = Path(config.get('save_path', './final_model.ckpt'))
    trainer.save_checkpoint(save_path)
    print(f"Final model saved to: {save_path}")


def test_model_creation(config_path: str):
    """Test function to verify model creation and basic forward pass"""
    print("Running model creation test...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock vocab size for testing
    test_vocab_size = 10000
    
    try:
        # Test model creation
        model_type = config['model']['type']
        model_config = config['model']['config']
        
        print(f"Testing {model_type} model creation...")
        model = ModelFactory.create_model(model_type, model_config, test_vocab_size)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = model_config.get('seq_len', 128)
        
        # Create dummy input (seq_len, batch_size) format
        dummy_input = torch.randint(0, test_vocab_size, (seq_len, batch_size))
        dummy_targets = torch.randint(0, test_vocab_size, (seq_len, batch_size))
        
        print(f"Testing forward pass with input shape: {dummy_input.shape}")
        
        model.eval()
        with torch.no_grad():
            if model_type.upper() == 'CBR_RNN':
                cache = model.init_cache(dummy_input)
                output, _ = model(dummy_input, initial_cache=cache)
                print(f"CBR_RNN output shape: {output.shape}")
                
            elif model_type.upper() == 'TRANSFORMER':
                # Transformer expects (batch, seq) format
                dummy_input_transposed = dummy_input.transpose(0, 1)
                output = model(dummy_input_transposed)
                print(f"Transformer output shape: {output.shape}")
                
            elif model_type.upper() == 'LSTM':
                # LSTM expects (batch, seq) format
                dummy_input_transposed = dummy_input.transpose(0, 1)
                output, _ = model(dummy_input_transposed)
                print(f"LSTM output shape: {output.shape}")
        
        print("Forward pass test successful!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal model training script')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--test-only', action='store_true', help='Only run model creation test')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--resume-comprehensive', type=str, default=None, 
                       help='Resume from comprehensive checkpoint (includes all training state)')
    
    args = parser.parse_args()
    
    if args.test_only:
        success = test_model_creation(args.config)
        exit(0 if success else 1)
    elif args.resume_comprehensive:
        # Resume from comprehensive checkpoint
        resume_training_from_comprehensive_checkpoint(args.resume_comprehensive, args.config)
    else:
        # Normal training or resume from standard Lightning checkpoint
        train_model(args.config, args.resume)