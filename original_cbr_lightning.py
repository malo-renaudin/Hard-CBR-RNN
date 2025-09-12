import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import datasets
import math
import numpy as np
from collections import Counter
from original_cbr import CueBasedRNNModel 
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
                 dropout=0.5, tie_weights=False, lr=1.0, weight_decay=0.0,
                 use_aux_objective=False, aux_vocab_size=0):
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
        self.model = CueBasedRNNModel(
            ntoken=vocab_size,
            ninp=ninp,
            nhid=nhid, 
            nlayers=nlayers,
            dropout=dropout,
            tie_weights=tie_weights,
            aux_objective=use_aux_objective,
            nauxclasses=aux_vocab_size,
            device=self.device
        )
        
        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # For auxiliary loss if needed
        if use_aux_objective:
            self.aux_criterion = nn.CrossEntropyLoss(reduction='mean')
            self.aux_weight = 0.1  # Weight for auxiliary loss
        
    def create_causal_mask(self, seq_len, batch_size, device):
        """
        Create causal attention masks for the sequence
        
        Args:
            seq_len: Length of sequence
            batch_size: Batch size
            device: Device to create mask on
            
        Returns:
            Causal mask of shape [batch_size, seq_len, seq_len+1]
        """
        # Create causal mask: position i can only attend to positions [0, 1, ..., i]
        mask = torch.full((seq_len, seq_len + 1), float('-inf'), device=device)
        
        # Fill lower triangular part with 0s (allowed positions)
        for i in range(seq_len):
            mask[i, :i+1] = 0.0
            
        # Expand for batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask
        
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
        
        # Create causal masks
        masks = self.create_causal_mask(seq_len, batch_size, sequences.device)
        
        # Forward pass through model
        output, final_hidden, aux_output, attn_log = self.model.forward(
            observation=sequences,
            initial_cache=initial_cache,
            masks=masks,
            attn_softmax_scaling_factor=1.0,
            output_attn=False,  # Set to True if you want to analyze attention
            uniform_attn=False,
            random_attn=False
        )
        
        # Compute primary language modeling loss
        # output: [seq_len, batch_size, vocab_size]
        # targets: [seq_len, batch_size]
        output_flat = output.reshape(-1, output.size(-1))  # [seq_len*batch_size, vocab_size]
        targets_flat = targets.reshape(-1)  # [seq_len*batch_size]
        
        lm_loss = self.criterion(output_flat, targets_flat)
        total_loss = lm_loss
        
        # Compute auxiliary loss if enabled
        aux_loss = None
        if self.hparams.use_aux_objective and aux_output is not None:
            aux_output_flat = aux_output.reshape(-1, aux_output.size(-1))
            # Note: You would need auxiliary targets for this to work
            # aux_loss = self.aux_criterion(aux_output_flat, aux_targets_flat)
            # total_loss = lm_loss + self.aux_weight * aux_loss
            pass
        
        # Compute perplexity for logging
        ppl = torch.exp(lm_loss)
        
        # Log metrics
        self.log(f"{stage}_loss", lm_loss, prog_bar=(stage == "train"), 
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ppl", ppl, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        
        if aux_loss is not None:
            self.log(f"{stage}_aux_loss", aux_loss, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_total_loss", total_loss, on_epoch=True, sync_dist=True)
        
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
        Configure optimizer and learning rate scheduler
        
        Uses SGD with momentum as in the reference training code
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler: reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            patience=2
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


def train_cbr_model():
    """
    Training script for the CueBasedRNNModel
    """
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    
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
    
    # Initialize model
    model = CBRLanguageModel(
        vocab_size=tokenizer.vocab_size,
        ninp=512,
        nhid=512,
        nlayers=1,
        dropout=0.5,
        tie_weights=False,
        lr=1.0,  # High LR as in reference
        weight_decay=0.0
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=0.25,  # Match reference implementation
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=500,
        val_check_interval=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    

    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_cbr_model()