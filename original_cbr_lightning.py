import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
import math
import numpy as np
from collections import Counter
from original_cbr import CueBasedRNNModel, OptimizedCueBasedRNNModel, OptimizedCueBasedRNNModel_MH, CBR_RNN
# Assume CueBasedRNNModel is imported from your module
# from your_module import CueBasedRNNModel

class WordTokenizer:
    """Simple word-level tokenizer for WikiText"""
    def __init__(self, list_of_texts, vocab_size=50000):
        tokens = []
        for text in list_of_texts:
            tokens.extend(text.split())
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)  # reserve for special tokens
        self.itos = ["<unk>"] + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(tok, 0) for tok in text.split()]  # 0 = <unk>

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])


class WikiTextDataset(Dataset):
    """Dataset for WikiText that produces sequences in the format expected by CueBasedRNNModel"""
    def __init__(self, dataset, tokenizer, seq_len=35):
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


class CBRLanguageModel(pl.LightningModule):
    """
    PyTorch Lightning module for training CueBasedRNNModel
    
    This module handles the training loop, optimization, and logging
    without modifying the original CueBasedRNNModel architecture.
    """
    
    def __init__(self, vocab_size, ninp=512, nhid=512, nlayers=1, 
                 dropout=0.5, nheads=8, lr=1.0, weight_decay=0,
                 use_gumbel_softmax=False, initial_temp=1.0, final_temp=0.1, temp_decay="exponential"):
        """
        Initialize the Lightning module
        
        Args:
            vocab_size: Size of vocabulary
            ninp: Embedding dimension
            nhid: Hidden dimension  
            nlayers: Number of layers (passed to model but not used in current implementation)
            dropout: Dropout probability
            tie_weights: Whether to tie input/output embeddings
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            use_aux_objective: Whether to use auxiliary objective
            aux_vocab_size: Size of auxiliary vocabulary
        """
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Initialize the CueBasedRNNModel
        self.model = CBR_RNN(
            ntoken=vocab_size,
            ninp=ninp,
            nhid=nhid, 
            nlayers=nlayers,
            dropout=dropout,
            nheads=nheads
        )
        
        # Gumbel softmax and temperature scheduling parameters
        self.use_gumbel_softmax = use_gumbel_softmax
        self.initial_temp = initial_temp if use_gumbel_softmax else None
        self.final_temp = final_temp if use_gumbel_softmax else None
        self.temp_decay = temp_decay if use_gumbel_softmax else None
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
    def get_current_temperature(self):
        """Calculate current temperature based on training progress"""
        if not self.use_gumbel_softmax:
            return 1.0  # Default temperature for regular softmax (not used)
            
        if not hasattr(self.trainer, 'max_epochs') or self.trainer.max_epochs == 0:
            return self.initial_temp
            
        progress = self.current_epoch / self.trainer.max_epochs
        
        if self.temp_decay == "linear":
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.temp_decay == "exponential":
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.temp_decay == "cosine":
            # Cosine annealing - smooth decay
            temp = self.final_temp + (self.initial_temp - self.final_temp) * \
                   0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Default to constant temperature
            temp = self.initial_temp
            
        return max(temp, 0.1)
  
        
    def _shared_step(self, batch, stage):
        """
        Shared step for train/val/test
        
        Args:
            batch: Batch of data (sequences, targets)
            stage: One of 'train', 'val', 'test'
            
        Returns:
            loss: Computed loss value
        """
        sequences, targets = batch
        batch_size, seq_len = sequences.shape
        
        # Transpose to match model expectation: [seq_len, batch_size]
        sequences = sequences.transpose(0, 1)
        targets = targets.transpose(0, 1)
        
        # Initialize cache for this batch
        initial_cache = self.model.init_cache(sequences)
        
        forward_kwargs = {
            'observation': sequences,
            'initial_cache': initial_cache
        }
        
        # Add temperature and gumbel flag only if using Gumbel softmax
        if self.use_gumbel_softmax:
            current_temp = self.get_current_temperature()
            forward_kwargs.update({
                'temperature': current_temp,
                'use_gumbel': True  # Assuming your model has this flag
            })
        # Forward pass through model
        output, final_hidden = self.model.forward(**forward_kwargs)
        
        # Compute primary language modeling loss
        # output: [seq_len, batch_size, vocab_size]
        # targets: [seq_len, batch_size]
        output_flat = output.reshape(-1, output.size(-1))  # [seq_len*batch_size, vocab_size]
        targets_flat = targets.reshape(-1)  # [seq_len*batch_size]
        
        lm_loss = self.criterion(output_flat, targets_flat)
        total_loss = lm_loss
      
        
        # Compute perplexity for logging
        ppl = torch.exp(lm_loss)
        
        # Log metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=(stage == "train"), 
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        
        # Only log temperature if using Gumbel softmax
        if self.use_gumbel_softmax:
            current_temp = self.get_current_temperature()
            self.log(f"{stage}_temperature", current_temp, prog_bar=(stage == "train"),
                    on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        return self._shared_step(batch, "test")
    
    def configure_optimizers(self):
        """
        Simple but effective optimizer for language modeling
        AdamW with cosine scheduling - proven to work well
        """
        # AdamW is the gold standard for transformers/language models
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),  # Standard values work well
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False
        )
        
        # Cosine annealing - smooth learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.01  # End at 1% of starting LR
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

def train_cbr_model(use_gumbel_softmax=False, gumbel_config=None, model_kwargs=model_kwargs):
    """
    Training script for the CueBasedRNNModel
    """
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if gumbel_config is None:
        gumbel_config = {
            'initial_temp': 1.0,
            'final_temp': 0.1,
            'temp_decay': 'exponential'
        }
    # Load WikiText-103 dataset
    data_dir = "cbr_lightning/wikitext-103-raw"  # Adjust path as needed
    raw = datasets.load_from_disk(data_dir)
    train_dataset = raw['train']
    val_dataset = raw['validation'] 
    test_dataset = raw['test']
    
    # Build tokenizer
    all_texts = list(train_dataset['text']) + list(val_dataset['text']) + list(test_dataset['text'])
    tokenizer = WordTokenizer(all_texts, vocab_size=50000)
    
    # Create datasets
    seq_len = 35  # Match reference implementation
    batch_size = 256
    
    train_ds = WikiTextDataset(train_dataset, tokenizer, seq_len)
    val_ds = WikiTextDataset(val_dataset, tokenizer, seq_len)
    test_ds = WikiTextDataset(test_dataset, tokenizer, seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True
    )
    
    if model_kwargs is None:
        model_kwargs = {
            'vocab_size': tokenizer.vocab_size,
            'ninp': 512,
            'nhid': 512,
            'nlayers': 1,
            'dropout': 0.5,
            'lr': 3e-4,
            'weight_decay': 0.0,
            'nheads': 8,
        }
    else:
        # Make sure vocab_size is set
        model_kwargs['vocab_size'] = tokenizer.vocab_size
    
    
    # Add Gumbel softmax parameters only if needed
    if use_gumbel_softmax:
        model_kwargs.update(gumbel_config)
        print(f"Training with Gumbel softmax: {gumbel_config}")
    else:
        print("Training with regular softmax")
    
    model = CBRLanguageModel(**model_kwargs)
    
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=5,
        gradient_clip_val=0.25,  # Match reference implementation
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=500,
        val_check_interval=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Allow non-deterministic ops for speed
        benchmark=True, 
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    

    
    return model, tokenizer


if __name__ == "__main__":
    
    model_kwargs = {
        'ninp': 512,
        'nhid': 512,
        'nlayers': 1,
        'dropout': 0.5,
        'lr': 3e-4,
        'weight_decay': 0.0,
        'nheads': 8,
    }
    model, tokenizer = train_cbr_model(use_gumbel_softmax=True)