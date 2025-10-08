"""Train a single CBR model with a given configuration"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets

from data_utils import WordTokenizer, WikiTextDataset
from lit_cbr import CBRLanguageModel


def train_single_job(job_id):
    """Train a single job given its ID"""
    # Create job directory
    job_dir = Path(f"job_cbr_2_{job_id:03d}")
    job_dir.mkdir(exist_ok=True)
    
    # Load config
    config_file = Path(f"job_cbr_2_configs/config_{job_id:03d}.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Save config to job directory
    with open(job_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Job {job_id}: Starting training with config: {config}")
    
    # Setup logging to file
    log_file = job_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load shared tokenizer
        tokenizer = WordTokenizer.load("shared_data/tokenizer.json")
        
        # Load data
        data_dir = "cbr_lightning/wikitext-103-raw"
        raw = datasets.load_from_disk(data_dir)
        
        # Create datasets
        train_ds = WikiTextDataset(raw['train'], tokenizer, seq_len=64)
        val_ds = WikiTextDataset(raw['validation'], tokenizer, seq_len=64)
        
        # Create data loaders
        train_loader = DataLoader(
            train_ds, batch_size=256, shuffle=True,
            num_workers=4, drop_last=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=256, shuffle=False,
            num_workers=4, drop_last=True, pin_memory=True
        )
        
        # Set up model kwargs
        model_kwargs = {
            'vocab_size': tokenizer.vocab_size,
            'ninp': config['nhid'],  # Use same as hidden
            'nhid': config['nhid'],
            'nlayers': 1,
            'dropout': config['dropout'],
            'nheads': config['nheads'],
            'lr': config['lr'],
            'weight_decay': 1e-4,
            'use_gumbel_softmax': config['use_gumbel_softmax']
        }
        
        # Add Gumbel parameters if needed
        if config['use_gumbel_softmax']:
            model_kwargs.update({
                'initial_temp': 1.0,
                'final_temp': config['final_temp'],
                'temp_decay': config['temp_decay']
            })
        
        # Create model
        model = CBRLanguageModel(**model_kwargs)
        
        # Setup trainer with job-specific checkpoint directory
        trainer = pl.Trainer(
            max_epochs=50,
            gradient_clip_val=0.25,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision='bf16-mixed' if torch.cuda.is_available() else 32,
            log_every_n_steps=500,
            val_check_interval=1.0,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=False,
            default_root_dir=str(job_dir),
            deterministic=False,
            benchmark=True
        )
        
        # Train the model
        logger.info(f"Starting training for job {job_id}")
        trainer.fit(model, train_loader, val_loader)
        
        # Save final results
        final_metrics = {
            'job_id': job_id,
            'config': config,
            'train_loss': float(trainer.callback_metrics.get('train_loss_epoch', float('inf'))),
            'val_loss': float(trainer.callback_metrics.get('val_loss_epoch', float('inf'))),
            'train_ppl': float(trainer.callback_metrics.get('train_ppl_epoch', float('inf'))),
            'val_ppl': float(trainer.callback_metrics.get('val_ppl_epoch', float('inf'))),
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(job_dir / "results.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"Job {job_id} completed successfully")
        logger.info(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        
        # Save error info
        error_info = {
            'job_id': job_id,
            'config': config,
            'status': f'failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(job_dir / "results.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_single.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    train_single_job(job_id)