#!/usr/bin/env python3
"""
Complete Training Script for CBR_RNN on WikiText-103
with Compression Cache Management, MLflow tracking, and Enhanced Diagnostics
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger
import argparse
import os
import time
import sys
from pathlib import Path
import json
import psutil
import gc
import mlflow
# Import your modules
from model_lightning import CBR_RNN, Transformer, LSTM
from wikitext_dataset import WikiTextDataModule
from utils import load_model

class ModelDiagnostics:
    """Class to handle model diagnostics and validation"""
    
    @staticmethod
    def validate_model_architecture(model, vocab_size, seq_len):
        """Validate model architecture and parameters"""
        print(f"\n{'='*60}")
        print("MODEL ARCHITECTURE VALIDATION")
        print(f"{'='*60}")
        
        # Basic parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Type: {type(model).__name__}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Parameter Memory: {total_params * 4 / 1024**2:.2f} MB (float32)")
        
        # Model-specific validations
        if isinstance(model, CBR_RNN):
            ModelDiagnostics._validate_cbr_rnn(model, vocab_size, seq_len)
        elif isinstance(model, Transformer):
            ModelDiagnostics._validate_transformer(model, vocab_size, seq_len)
        elif isinstance(model, LSTM):
            ModelDiagnostics._validate_lstm(model, vocab_size, seq_len)
        
        # Memory usage
        ModelDiagnostics._check_memory_usage()
        return True
    
    @staticmethod
    def _validate_cbr_rnn(model, vocab_size, seq_len):
        """Validate CBR_RNN specific architecture"""
        print(f"\nCBR_RNN Specific Validation:")
        print(f"  Vocabulary Size: {vocab_size} (expected: {model.hparams.ntoken})")
        print(f"  Sequence Length: {seq_len} (model seq_len: {model.seq_len})")
        print(f"  Compressed Dim: {model.compressed_dim}")
        print(f"  Compression Ratio: {model.compressed_dim/seq_len:.4f}")
        print(f"  Hidden Dim: {model.nhid}")
        print(f"  Attention Heads: {model.nheads}")
        print(f"  Gumbel Softmax: {model.gumbel_softmax}")
        print(f"  Temperature: {model.temperature}")
        
        # Validate compression layers
        expected_compress_input = model.nhid * seq_len
        actual_compress_input = model.hidden_compress.in_features
        print(f"  Compression layer input: {actual_compress_input} (expected: {expected_compress_input})")
        
        if actual_compress_input != expected_compress_input:
            raise ValueError(f"CBR_RNN compression layer dimension mismatch: "
                           f"expected {expected_compress_input}, got {actual_compress_input}")
    
    @staticmethod
    def _validate_transformer(model, vocab_size, seq_len):
        """Validate Transformer specific architecture"""
        print(f"\nTransformer Specific Validation:")
        print(f"  Vocabulary Size: {vocab_size} (model vocab_size: {model.vocab_size})")
        print(f"  Model Dimension: {model.d_model}")
        print(f"  Layers: {len(model.transformer_blocks)}")
        print(f"  Max Sequence Length: {model.hparams.max_seq_len}")
        
    @staticmethod
    def _validate_lstm(model, vocab_size, seq_len):
        """Validate LSTM specific architecture"""
        print(f"\nLSTM Specific Validation:")
        print(f"  Vocabulary Size: {vocab_size} (model vocab_size: {model.vocab_size})")
        print(f"  Embedding Dim: {model.embedding_dim}")
        print(f"  Hidden Dim: {model.hidden_dim}")
        print(f"  Num Layers: {model.num_layers}")
    
    @staticmethod
    def _check_memory_usage():
        """Check current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU Memory Allocated: {gpu_memory:.2f} MB")
            print(f"  GPU Memory Cached: {gpu_memory_cached:.2f} MB")
        
        cpu_memory = psutil.virtual_memory()
        print(f"  CPU Memory Usage: {cpu_memory.percent:.1f}% "
              f"({cpu_memory.used / 1024**2:.0f} MB / {cpu_memory.total / 1024**2:.0f} MB)")
    
    @staticmethod
    def test_forward_pass(model, data_module):
        """Test model forward pass with sample data"""
        print(f"\n{'='*60}")
        print("FORWARD PASS VALIDATION")
        print(f"{'='*60}")
        
        model.eval()
        with torch.no_grad():
            # Get a sample batch
            train_loader = data_module.train_dataloader()
            sample_batch = next(iter(train_loader))
            inputs, targets = sample_batch
            
            print(f"Input shape: {inputs.shape}")
            print(f"Target shape: {targets.shape}")
            print(f"Input dtype: {inputs.dtype}")
            print(f"Input device: {inputs.device}")
            print(f"Input range: [{inputs.min().item()}, {inputs.max().item()}]")
            
            # Test forward pass
            try:
                if isinstance(model, CBR_RNN):
                    cache = model.init_cache(inputs)
                    print(f"Initial cache shapes: {[tuple(c.shape) for c in cache]}")
                    outputs, new_cache = model(inputs, initial_cache=cache)
                    if new_cache is not None:
                        print(f"Output cache shapes: {[tuple(c.shape) for c in new_cache]}")
                elif isinstance(model, (Transformer, LSTM)):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                
                print(f"Output shape: {outputs.shape}")
                print(f"Output dtype: {outputs.dtype}")
                print(f"Output finite: {torch.isfinite(outputs).all().item()}")
                print(f"Output range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                
                # Test loss calculation
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                print(f"Sample loss: {loss.item():.6f}")
                print(f"Sample perplexity: {torch.exp(loss).item():.2f}")
                
                print("✅ Forward pass validation PASSED")
                return True
                
            except Exception as e:
                print(f"❌ Forward pass validation FAILED: {e}")
                import traceback
                traceback.print_exc()
                return False
        
            finally:
                model.train()

class MLflowCallback(pl.Callback):
    """Custom callback for MLflow logging"""
    
    def on_train_start(self, trainer, pl_module):
        # Log model architecture details
        if hasattr(pl_module, 'hparams'):
            for key, value in pl_module.hparams.items():
                mlflow.log_param(f"model_{key}", value)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Log additional metrics
        if hasattr(pl_module, 'epoch_cache') and pl_module.epoch_cache is not None:
            cache_memory = sum(c.numel() * 4 for c in pl_module.epoch_cache) / 1024**2  # MB
            mlflow.log_metric("cache_memory_mb", cache_memory, step=trainer.current_epoch)
        
        # Log GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            mlflow.log_metric("gpu_memory_mb", gpu_memory, step=trainer.current_epoch)

def create_mlflow_logger(args):
    """Create MLflow logger with proper experiment setup"""
    
    # Set MLflow tracking URI if specified
    if hasattr(args, 'mlflow_tracking_uri') and args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Create or get experiment
    experiment_name = f"wikitext_{args.model.lower()}_{args.experiment_name}"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    # Create run name with key parameters
    run_name = (f"{args.model}_seq{args.sequence_length}_batch{args.batch_size}_"
                f"lr{args.learning_rate}_nhid{args.nhid}")
    
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "model_type": args.model,
            "dataset": "wikitext-103",
            "sequence_length": str(args.sequence_length),
            "batch_size": str(args.batch_size),
        }
    )
    
    return mlflow_logger

def create_callbacks(args):
    """Create PyTorch Lightning callbacks"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
        filename=f'{args.model}-{{epoch:02d}}-{{val_loss:.3f}}-{{val_ppl:.2f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=False,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=args.early_stopping_delta,
            patience=args.patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Custom MLflow callback
    callbacks.append(MLflowCallback())
    
    return callbacks

def create_loggers(args):
    """Create loggers for training"""
    loggers = []
    
    # MLflow logger (primary)
    mlflow_logger = create_mlflow_logger(args)
    loggers.append(mlflow_logger)
    
    # TensorBoard logger (backup)
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=f"{args.model}_v{args.vocab_size}_{args.sequence_length}_{args.batch_size}"
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy analysis
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name=f"{args.experiment_name}_csv"
    )
    loggers.append(csv_logger)
    
    return loggers

def create_trainer(args):
    """Create PyTorch Lightning trainer"""
    
    callbacks = create_callbacks(args)
    loggers = create_loggers(args)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.clip_grad,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=args.deterministic,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=not args.no_checkpointing
    )
    
    return trainer

def print_training_summary(args):
    """Print comprehensive training configuration"""
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Precision: {args.precision}")
    
    print(f"\nData Configuration:")
    print(f"  Vocabulary Size: {args.vocab_size:,}")
    print(f"  Sequence Length: {args.sequence_length}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Num Workers: {args.num_workers}")
    
    print(f"\nModel Configuration:")
    print(f"  Embedding Dim: {args.ninp}")
    print(f"  Hidden Dim: {args.nhid}")
    print(f"  Attention Heads: {args.nheads}")
    print(f"  Dropout: {args.dropout}")
    if args.model == 'CBR_RNN':
        print(f"  Compressed Dim: {args.compressed_dim}")
        print(f"  Gumbel Softmax: {args.gumbel_softmax}")
        print(f"  Temperature: {args.temperature}")
    
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Optimizer: {args.optimizer_type}")
    print(f"  Scheduler: {args.scheduler_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Gradient Clipping: {args.clip_grad}")
    print(f"  Accumulate Grad Batches: {args.accumulate_grad_batches}")
    
    print(f"\nLogging Configuration:")
    print(f"  Log Directory: {args.log_dir}")
    print(f"  Checkpoint Directory: {args.checkpoint_dir}")
    print(f"  Log Every N Steps: {args.log_every_n_steps}")
    print(f"  Validation Check Interval: {args.val_check_interval}")
    
    if args.early_stopping:
        print(f"\nEarly Stopping:")
        print(f"  Patience: {args.patience}")
        print(f"  Min Delta: {args.early_stopping_delta}")
    
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Train models on WikiText-103 with MLflow tracking')
    
    # Data parameters
    parser.add_argument('--model', type=str, default='CBR_RNN', 
                        choices=['CBR_RNN', 'Transformer', 'LSTM'])
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--sequence_length', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_val_samples', type=int, default=2000)
    parser.add_argument('--tokenizer_path', type=str, default = './tokenizer.pkl')
    parser.add_argument('--tokenized_path', type=str, default = 'cbr_lightning/wikitext-103-tokenized')
    
    # Model architecture parameters
    parser.add_argument('--ninp', type=int, default=512)
    parser.add_argument('--nhid', type=int, default=1024)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--compressed_dim', type=int, default=1)
    
    # Gumbel Softmax parameters
    parser.add_argument('--gumbel_softmax', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
    
    # Training setup
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    # MLflow configuration
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None,
                       help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment', type=str, default='wikitext_experiments',
                       help='MLflow experiment name')
    
    # Logging and checkpointing
    parser.add_argument('--experiment_name', type=str, default='model_comparison')
    parser.add_argument('--log_dir', type=str, default='lightning_logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--early_stopping_delta', type=float, default=0.001)
    
    # Testing/debugging
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--no_checkpointing', action='store_true')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    # Validation flags
    parser.add_argument('--skip_validation', action='store_true', help='Skip model validation')
    parser.add_argument('--skip_forward_test', action='store_true', help='Skip forward pass test')
    
    args = parser.parse_args()
    
    # Print configuration
    print_training_summary(args)
    
    # Set random seed for reproducibility
    if args.deterministic:
        pl.seed_everything(42, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    try:
        # Set up data module with enhanced validation
        print(f"\n{'='*60}")
        print("DATA MODULE SETUP")
        print(f"{'='*60}")
        
        # Load tokenizer first
        print(f"Loading tokenizer from: {args.tokenizer_path}")
        import pickle
        with open(args.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Handle different tokenizer formats
        if isinstance(tokenizer, dict):
            vocab_size = len(tokenizer['word2idx'])
        else:
            vocab_size = len(tokenizer.word2idx) if hasattr(tokenizer, 'word2idx') else getattr(tokenizer, 'vocab_size', None)
        
        print(f"✅ Tokenizer loaded. Vocabulary size: {vocab_size:,}")
        
        # Create data module
        data_module = WikiTextDataModule(
            tokenized_path=args.tokenized_path,
            tokenizer=tokenizer,
            sequence_length=128,
            stride=32,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Setup data module
        data_module.setup('fit')
        
        print(f"✅ Training samples: {len(data_module.train_dataset):,}")
        print(f"✅ Validation samples: {len(data_module.val_dataset):,}")
        
        # Run sanity checks if available
        if hasattr(data_module, 'run_sanity_checks'):
            print(f"\nRunning data module sanity checks...")
            try:
                data_module.run_sanity_checks()
            except Exception as e:
                print(f"⚠️ Sanity checks failed: {e}")
                print("Continuing without sanity checks...")
        
        # Validate data integrity
        print(f"\nValidating data integrity...")
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        inputs, targets = batch
        
        print(f"✅ Input shape: {inputs.shape}")
        print(f"✅ Target shape: {targets.shape}")
        print(f"✅ Input range: [{inputs.min().item()}, {inputs.max().item()}]")
        print(f"✅ Target range: [{targets.min().item()}, {targets.max().item()}]")
        
        # Check for out-of-bounds tokens
        if inputs.max().item() >= vocab_size or inputs.min().item() < 0:
            raise ValueError(f"Input tokens out of bounds: range=[{inputs.min().item()}, {inputs.max().item()}], vocab_size={vocab_size}")
        if targets.max().item() >= vocab_size or targets.min().item() < 0:
            raise ValueError(f"Target tokens out of bounds: range=[{targets.min().item()}, {targets.max().item()}], vocab_size={vocab_size}")
        
        print("✅ All token IDs are within vocabulary bounds")
        
        # Create model
        print(f"\n{'='*60}")
        print("MODEL CREATION")
        print(f"{'='*60}")
        
        model = load_model(args, vocab_size)
        print(f"✅ Model created: {type(model).__name__}")
        
        # Model validation
        if not args.skip_validation:
            ModelDiagnostics.validate_model_architecture(model, vocab_size, args.sequence_length)
        
        # Forward pass test
        if not args.skip_forward_test:
            success = ModelDiagnostics.test_forward_pass(model, data_module)
            if not success:
                print("❌ Forward pass test failed. Aborting training.")
                sys.exit(1)
        
        # Create trainer
        trainer = create_trainer(args)
        
        # Log initial system state
        print(f"\n{'='*60}")
        print("SYSTEM STATE")
        print(f"{'='*60}")
        ModelDiagnostics._check_memory_usage()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{args.model}_{int(time.time())}"):
            # Log all arguments as parameters
            for key, value in vars(args).items():
                mlflow.log_param(key, value)
            
            # Log system info
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("torch_version", torch.__version__)
            mlflow.log_param("lightning_version", pl.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name())
            
            # Train model
            print(f"\n{'='*80}")
            print("STARTING TRAINING")
            print(f"{'='*80}")
            start_time = time.time()
            
            if args.resume_from_checkpoint:
                print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
                trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
            else:
                trainer.fit(model, data_module)
            
            training_time = time.time() - start_time
            
            # Log final metrics
            mlflow.log_metric("training_time_minutes", training_time / 60)
            mlflow.log_metric("final_train_loss", trainer.callback_metrics.get('train_loss', float('inf')))
            mlflow.log_metric("final_val_loss", trainer.callback_metrics.get('val_loss', float('inf')))
            
            print(f"\n{'='*80}")
            print("TRAINING COMPLETED")
            print(f"{'='*80}")
            print(f"Training time: {training_time/60:.2f} minutes")
            print(f"Final train loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")
            print(f"Final validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
            print(f"Final validation perplexity: {trainer.callback_metrics.get('val_ppl', 'N/A')}")
            
            # Save final model
            if not args.no_checkpointing:
                final_model_path = os.path.join(args.checkpoint_dir, args.experiment_name, f'final_{args.model}.ckpt')
                trainer.save_checkpoint(final_model_path)
                mlflow.log_artifact(final_model_path)
                print(f"Final model saved and logged to MLflow: {final_model_path}")
            
            print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*50}")
        print("Training interrupted by user")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"Training failed with error: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()