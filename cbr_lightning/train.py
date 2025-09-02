#!/usr/bin/env python3
"""
Complete Training Script for CBR_RNN on WikiText-103
with Compression Cache Management
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
import os
import time
import sys

# Import your modules
from model_lightning import CBR_RNN
from wikitext_dataset import WikiTextDataModule
from utils import load_model

def create_callbacks(args):
    """Create PyTorch Lightning callbacks"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
        filename='{epoch:02d}-{val_loss:.3f}-{val_ppl:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=False
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
    
    return callbacks

def create_loggers(args):
    """Create loggers for training"""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=f"v{args.vocab_size}_{args.sequence_length}_{args.batch_size}"
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

def print_model_info(model, data_module):
    """Print detailed model and data information"""
    print("\n" + "="*80)
    print("MODEL & DATA CONFIGURATION")
    print("="*80)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Embedding Dim: {model.hparams.ninp}")
    print(f"  Hidden Dim: {model.hparams.nhid}")
    print(f"  Attention Heads: {model.hparams.nheads}")
    print(f"  Vocabulary Size: {model.hparams.ntoken}")
    
    
    # Gumbel Softmax info
    print(f"\nAttention Configuration:")
    print(f"  Gumbel Softmax: {model.gumbel_softmax}")
    print(f"  Temperature: {model.temperature}")
    
    # Data info
    print(f"\nDataset Information:")
    print(f"  Vocab Size: {len(data_module.tokenizer.word2idx)}")
    print(f"  Sequence Length: {data_module.sequence_length}")
    print(f"  Batch Size: {data_module.batch_size}")
    print(f"  Train Dataset: {len(data_module.train_dataset):,} sequences")
    print(f"  Val Dataset: {len(data_module.val_dataset):,} sequences")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Train CBR_RNN on WikiText-103')
    
    # Data parameters
    parser.add_argument('--model', type = str, default = 'CBR_RNN', 
                        choices = ['CBR_RNN', 'Transformer', 'LSTM'])
    parser.add_argument('--vocab_size', type=int, default=50000, 
                       help='Vocabulary size')
    parser.add_argument('--sequence_length', type=int, default=35, 
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size')
    parser.add_argument('--max_val_samples', type=int, default=2000, 
                       help='Limit validation samples for faster validation')
    
    # Model architecture parameters
    parser.add_argument('--ninp', type=int, default=512, 
                       help='Embedding dimension')
    parser.add_argument('--nhid', type=int, default=1024, 
                       help='Hidden dimension')
    parser.add_argument('--nheads', type=int, default=1, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    parser.add_argument('--compressed_dim', type=int, default=1, help='size of the sequence length dimension in the compressed cache')
    
    # Gumbel Softmax parameters
    parser.add_argument('--gumbel_softmax', action='store_true', 
                       help='Use Gumbel Softmax instead of regular softmax')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Temperature for Gumbel Softmax')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--optimizer_type', type=str, default='adamw', 
                       choices=['adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, default='cosine', 
                       choices=['cosine', 'step', 'plateau'], help='Learning rate scheduler')
    # Training setup
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of epochs')
    parser.add_argument('--devices', type=int, default=1, 
                       help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='16-mixed', 
                       help='Training precision')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                       help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, 
                       help='Accumulate gradients over N batches')
    
    # Logging and checkpointing
    parser.add_argument('--experiment_name', type=str, default='cbr_rnn_wikitext', 
                       help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='lightning_logs', 
                       help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory for checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=1000, 
                       help='Log every N steps')
    parser.add_argument('--val_check_interval', type=float, default=1.0, 
                       help='Validation check interval (epochs)')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', 
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=7, 
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement')
    
    # Testing/debugging
    parser.add_argument('--limit_train_batches', type=float, default=10000, 
                       help='Limit training batches (for testing)')
    parser.add_argument('--limit_val_batches', type=float, default=1, 
                       help='Limit validation batches (for testing)')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of dataloader workers')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run 1 batch of train/val/test for debugging')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic training for reproducibility')
    parser.add_argument('--no_checkpointing', action='store_true',
                       help='Disable checkpointing (for debugging)')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("CBR_RNN Training on WikiText-103")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Data: vocab_size={args.vocab_size}, seq_len={args.sequence_length}, batch_size={args.batch_size}")
    print(f"Model: ninp={args.ninp}, nhid={args.nhid}, nheads={args.nheads}")
    print(f"Attention: gumbel={args.gumbel_softmax}, temp={args.temperature}")
    print(f"Training: lr={args.learning_rate}, epochs={args.epochs}, devices={args.devices}")
    print("="*80)
    
    # Set random seed for reproducibility
    if args.deterministic:
        pl.seed_everything(42, workers=True)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set up data module
    print("Setting up data module...")
    data_module = WikiTextDataModule(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        #max_val_samples=args.max_val_samples
    )

    data_module.prepare_data()
    data_module.setup()
    print(f"Vocabulary size: {len(data_module.tokenizer.word2idx)}")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print("\n--- Debugging batch data ---")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    inputs, targets = batch
    
    vocab_size = len(data_module.tokenizer.word2idx)
    print(f"Input shape: {inputs.shape}")  # Should be (seq_len, batch_size)
    print(f"Target shape: {targets.shape}")  # Should be (seq_len, batch_size)
    print(f"Input min/max: {inputs.min().item()}/{inputs.max().item()}")
    print(f"Target min/max: {targets.min().item()}/{targets.max().item()}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Validate all token IDs are within bounds
    if inputs.max().item() >= vocab_size or inputs.min().item() < 0:
        raise ValueError(f"Input tokens out of bounds: min={inputs.min().item()}, max={inputs.max().item()}, vocab_size={vocab_size}")
    if targets.max().item() >= vocab_size or targets.min().item() < 0:
        raise ValueError(f"Target tokens out of bounds: min={targets.min().item()}, max={targets.max().item()}, vocab_size={vocab_size}")
    
    print("✓ All token IDs are within vocabulary bounds")
    
    # Set up the data to get vocabulary size
    data_module.setup()
    vocab_size = len(data_module.tokenizer.word2idx)
    
    # Create model
    print("Creating model...")
    # model = CBR_RNN(
    #     ntoken=vocab_size,
    #     ninp=args.ninp,
    #     nhid=args.nhid,
    #     nheads=args.nheads,
    #     dropout=args.dropout,
    #     learning_rate=args.learning_rate,
    #     temperature=args.temperature,
    #     gumbel_softmax=args.gumbel_softmax,
    #     criterion='cross_entropy',
    #     optimizer_type=args.optimizer_type,
    #     weight_decay=args.weight_decay,
    #     scheduler_type=args.scheduler_type,
    #     seq_len = args.sequence_length,
    #     compressed_dim = args.compressed_dim,
    # )
    model = load_model(args, vocab_size)

    # Print model and data info
    # print_model_info(model, data_module)
    
    # Create trainer
    trainer = create_trainer(args)
    
    # Quick validation before training
    if not args.fast_dev_run:
        print("\nRunning quick validation check...")
        try:
            # Test one batch
            model.eval()
            val_loader = data_module.val_dataloader()
            test_batch = next(iter(val_loader))
            with torch.no_grad():
                test_output = model.validation_step(test_batch, 0)
            print(f"✅ Validation check passed - loss: {test_output:.4f}")
        except Exception as e:
            print(f"❌ Validation check failed: {e}")
            sys.exit(1)
    
    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    start_time = time.time()
    
    try:
        if args.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, data_module)
        
        training_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Best validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
        print(f"Best validation perplexity: {trainer.callback_metrics.get('val_ppl', 'N/A')}")
        
        # Save final model
        if not args.no_checkpointing:
            final_model_path = os.path.join(args.checkpoint_dir, args.experiment_name, 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            print(f"Final model saved to: {final_model_path}")
        
        # Print cache statistics if persistent cache was used
#        if args.persistent_cache and hasattr(model, '_persistent_cache') and model._persistent_cache is not None:
#            final_cache_length = model._persistent_cache[1].size(1)
#            print(f"Final cache length: {final_cache_length}")
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("Training interrupted by user")
        print("="*50)
        
    except Exception as e:
        print(f"\n" + "="*50)
        print(f"Training failed with error: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
