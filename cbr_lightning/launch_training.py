#!/usr/bin/env python3
"""
Universal Training Script for CBR_RNN, Transformer, and LSTM models
with PyTorch Lightning and MLflow integration
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import datasets
import pickle

# Import your models
from model_lightning import CBR_RNN, Transformer, LSTM


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


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create PyTorch Lightning callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config.get('dirpath', './checkpoints'),
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
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


class TemperatureSchedulerCallback(pl.Callback):
    """Callback to update temperature scheduler every epoch"""
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Update temperature at the start of each training epoch"""
        if hasattr(pl_module, 'temp_scheduler') and hasattr(pl_module, 'gumbel_softmax'):
            if pl_module.gumbel_softmax:
                pl_module.temp_scheduler.step()
                current_temp = pl_module.temp_scheduler.get_temperature()
                pl_module.temperature = current_temp
                
                # Log temperature
                if trainer.logger:
                    trainer.logger.log_metrics({'temperature': current_temp}, step=trainer.global_step)


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
    
    # Create logger
    logger_config = config.get('logging', {})
    logger = MLFlowLogger(
        experiment_name=logger_config.get('experiment_name', 'language_modeling'),
        tracking_uri=logger_config.get('tracking_uri', './mlruns'),
        tags={'model_type': model_type, 'total_params': total_params}
    )
    
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
    
    logger.log_hyperparams(hyperparams)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Add temperature scheduler callback if needed
    if hasattr(model, 'temp_scheduler'):
        callbacks.append(TemperatureSchedulerCallback())
    
    # Create trainer
    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        logger=logger,
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


def create_example_config():
    """Create example configuration files for different models"""
    
    configs = {
        'cbr_rnn_config.yaml': {
            'seed': 42,
            'model': {
                'type': 'CBR_RNN',
                'config': {
                    'ninp': 256,
                    'nhid': 512,
                    'nheads': 4,
                    'seq_len': 128,
                    'compressed_dim': 32,
                    'dropout': 0.1,
                    'learning_rate': 1e-3,
                    'temperature': 1.0,
                    'gumbel_softmax': False,
                    'criterion': 'cross_entropy'
                }
            },
            'data': {
                'dataset_path': 'cbr_lightning/wikitext-103-tokenized',
                'tokenizer_path': './tokenizer.pkl',
                'batch_size': 32,
                'num_workers': 4,
                'max_length': 128
            },
            'trainer': {
                'max_epochs': 50,
                'accelerator': 'auto',
                'devices': 'auto',
                'precision': '16-mixed',
                'gradient_clip_val': 0.5,
                'accumulate_grad_batches': 1,
                'val_check_interval': 0.5
            },
            'logging': {
                'experiment_name': 'CBR_RNN_experiments',
                'tracking_uri': './mlruns'
            },
            'checkpoint': {
                'dirpath': './checkpoints/cbr_rnn',
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3
            },
            'early_stopping': {
                'enabled': True,
                'monitor': 'val_loss',
                'patience': 10,
                'mode': 'min'
            },
            'run_test': True,
            'save_path': './final_models/cbr_rnn_final.ckpt'
        },
        
        'transformer_config.yaml': {
            'seed': 42,
            'model': {
                'type': 'Transformer',
                'config': {
                    'ninp': 384,  # d_model
                    'nheads': 8,
                    'n_layers': 6,
                    'd_ff': 1536,
                    'seq_len': 128,  # max_seq_len
                    'dropout': 0.1,
                    'learning_rate': 1e-3,
                    'temperature': 1.0,
                    'gumbel_softmax': False
                }
            },
            'data': {
                'dataset_path': 'cbr_lightning/wikitext-103-tokenized',
                'tokenizer_path': './tokenizer.pkl',
                'batch_size': 32,
                'num_workers': 4,
                'max_length': 128
            },
            'trainer': {
                'max_epochs': 50,
                'accelerator': 'auto',
                'devices': 'auto',
                'precision': '16-mixed',
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1,
                'val_check_interval': 0.5
            },
            'logging': {
                'experiment_name': 'Transformer_experiments',
                'tracking_uri': './mlruns'
            },
            'checkpoint': {
                'dirpath': './checkpoints/transformer',
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3
            },
            'early_stopping': {
                'enabled': True,
                'monitor': 'val_loss',
                'patience': 10,
                'mode': 'min'
            },
            'run_test': True,
            'save_path': './final_models/transformer_final.ckpt'
        },
        
        'lstm_config.yaml': {
            'seed': 42,
            'model': {
                'type': 'LSTM',
                'config': {
                    'ninp': 256,  # embedding_dim
                    'nhid': 512,  # hidden_dim
                    'learning_rate': 1e-3
                }
            },
            'data': {
                'dataset_path': 'cbr_lightning/wikitext-103-tokenized',
                'tokenizer_path': './tokenizer.pkl',
                'batch_size': 32,
                'num_workers': 4,
                'max_length': 128
            },
            'trainer': {
                'max_epochs': 50,
                'accelerator': 'auto',
                'devices': 'auto',
                'precision': '16-mixed',
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1,
                'val_check_interval': 0.5
            },
            'logging': {
                'experiment_name': 'LSTM_experiments',
                'tracking_uri': './mlruns'
            },
            'checkpoint': {
                'dirpath': './checkpoints/lstm',
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3
            },
            'early_stopping': {
                'enabled': True,
                'monitor': 'val_loss',
                'patience': 10,
                'mode': 'min'
            },
            'run_test': True,
            'save_path': './final_models/lstm_final.ckpt'
        }
    }
    
    # Save example configs
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Example config created: {filename}")


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