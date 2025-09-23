import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
import math
import numpy as np
from collections import Counter
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from entire_transformer import SimpleTransformerLM

class WordTokenizer:
    """Simple word-level tokenizer for WikiText"""
    def __init__(self, list_of_texts, vocab_size=50000):
        tokens = []
        for text in list_of_texts:
            tokens.extend(text.split())
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)
        self.itos = ["<unk>"] + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(tok, 0) for tok in text.split()]

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])

    def save(self, path):
        """Save tokenizer to file"""
        with open(path, 'w') as f:
            json.dump({
                'itos': self.itos,
                'stoi': self.stoi,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls.__new__(cls)
        tokenizer.itos = data['itos']
        tokenizer.stoi = data['stoi']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer


class WikiTextDataset(Dataset):
    """Dataset for WikiText that produces sequences in the format expected by CueBasedRNNModel"""
    def __init__(self, dataset, tokenizer, seq_len=64):
        self.seq_len = seq_len
        text = " ".join(list(dataset["text"]))
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.data[i:i+self.seq_len]
        target = self.data[i+1:i+self.seq_len+1]
        return seq, target





def create_configs():
    """Create all configuration files for job array"""
    configs = [
        {'d_model': 256, 'nhead': 1, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': False},
        {'d_model': 1024, 'nhead': 1, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': False},
        {'d_model': 256, 'nhead': 8, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': False},
        {'d_model': 1024, 'nhead': 8, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': False},
        {'d_model': 256, 'nhead': 1, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': True},
        {'d_model': 1024, 'nhead': 1, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': True},
        {'d_model': 256, 'nhead': 8, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': True},
        {'d_model': 1024, 'nhead': 8, 'lr': 5e-4, 'dropout': 0.5, 'use_gumbel_softmax': True},
    ]
    
    # Create configs directory
    configs_dir = Path("job_transformer_2_configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Save each config
    for i, config in enumerate(configs):
        config_file = configs_dir / f"config_{i:03d}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create SLURM job script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=transformer_training
#SBATCH --array=0-{len(configs)-1}
#SBATCH --output=job_outputs/job_transformer_2_%A_%a.out
#SBATCH --error=job_outputs/job_transformer_2_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --constraint=h100
#SBATCH --account=ywa@h100
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu_p6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malorenaudin1@gmail.com
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=SIGUSR1@90

# Create output directory
mkdir -p job_outputs

# Run the training script with the config for this array job
python grid_search_transformer.py $SLURM_ARRAY_TASK_ID
"""
    
    with open("run_array_transformer.sh", 'w') as f:
        f.write(slurm_script)
    
    print(f"Created {len(configs)} config files and SLURM script")
    return len(configs)


def prepare_shared_data():
    """Prepare and save shared data (tokenizer and datasets) once"""
    print("Preparing shared data...")
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Load WikiText-103 dataset
    data_dir = "cbr_lightning/wikitext-103-raw"
    raw = datasets.load_from_disk(data_dir)
    train_dataset = raw['train']
    val_dataset = raw['validation']
    
    # Build tokenizer
    all_texts = list(train_dataset['text']) + list(val_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=50000)
    
    # Save tokenizer
    shared_dir = Path("shared_data")
    shared_dir.mkdir(exist_ok=True)
    tokenizer.save(shared_dir / "tokenizer.json")
    
    print(f"Saved tokenizer with vocab size: {tokenizer.vocab_size}")
    return tokenizer.vocab_size


def train_single_job(job_id):
    """Train a single job given its ID"""
    # Create job directory
    job_dir = Path(f"job_transformer_2_{job_id:03d}")
    job_dir.mkdir(exist_ok=True)
    
    # Load config
    config_file = Path(f"job_transformer_2_configs/config_{job_id:03d}.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Save config to job directory
    with open(job_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Job {job_id}: Starting training with config: {config}")
    
    # Setup logging to file
    import logging
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
            'd_model': config['d_model'],  
            'num_layers': 2,
            'dropout': config['dropout'],
            'nhead': config['nhead'],
            'lr': config['lr'],
            'weight_decay': 1e-4,
            'use_gumbel_softmax': config['use_gumbel_softmax']
        }
        
        # Add Gumbel parameters if needed
        if config['use_gumbel_softmax']:
            model_kwargs.update({
                'initial_temp': 1.0,
                'final_temp': 0.1,
                'temp_decay': 'exponential'
            })
        
        # Create model
        model = SimpleTransformerLM(**model_kwargs)
      
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


def collect_results():
    """Collect results from all completed jobs"""
    results = []
    
    for job_dir in Path(".").glob("job_*"):
        results_file = job_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results.append(json.load(f))
    
    # Sort by validation loss
    completed_results = [r for r in results if r['status'] == 'completed']
    completed_results.sort(key=lambda x: x['val_loss'])
    
    print(f"\nCollected {len(results)} total results ({len(completed_results)} successful)")
    print("\nTop 5 results:")
    for i, result in enumerate(completed_results[:5]):
        print(f"{i+1}. Job {result['job_id']:03d} - Val Loss: {result['val_loss']:.4f} - Config: {result['config']}")
    
    # Save consolidated results
    with open("all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Setup mode - create configs and prepare data
        print("Setting up parallel training...")
        vocab_size = prepare_shared_data()
        num_configs = create_configs()
        
        print(f"""
Setup complete!

To run the job array:
1. Submit the job: sbatch _transformer.sh
2. Monitor jobs: squeue -u $USER
3. Check results: python train_single.py collect

Files created:
- job_transformer_2_configs/config_*.json ({num_configs} config files)
- shared_data/tokenizer.json
- run_array_transformer.sh (SLURM script)
        """)
        
    elif len(sys.argv) == 2:
        if sys.argv[1] == "collect":
            # Collect results mode
            collect_results()
        else:
            # Single job mode
            job_id = int(sys.argv[1])
            train_single_job(job_id)