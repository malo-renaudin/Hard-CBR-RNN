 #!/usr/bin/env python3
"""
Train all models with a standard PyTorch loop, using configs.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# --- import your code ---
from cbr_lightning.models import CBR_RNN, Transformer, LSTM  # adjust imports
from cbr_lightning.data import get_dataset  # adjust to your dataset loader

# -----------------------
# Helper functions
# -----------------------

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(config):
    model_type = config['model']['type']
    model_cfg = config['model']['config']
    
    if model_type == "CBR_RNN":
        return CBR_RNN(**model_cfg)
    elif model_type == "Transformer":
        return Transformer(**model_cfg)
    elif model_type == "LSTM":
        return LSTM(**model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_dataloaders(config):
    data_cfg = config['data']
    train_ds = get_dataset(data_cfg['dataset_path'], split='train', max_length=data_cfg['max_length'])
    val_ds = get_dataset(data_cfg['dataset_path'], split='val', max_length=data_cfg['max_length'])
    
    train_loader = DataLoader(train_ds, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=data_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers'])
    return train_loader, val_loader

def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, criterion, dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), ignore_index=-100)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(dataloader.dataset)

# -----------------------
# Main training loop
# -----------------------

def main(config_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dir = Path(config_dir)
    config_files = list(config_dir.glob("*.yaml"))

    for cfg_file in config_files:
        print(f"\n=== Training {cfg_file.name} ===")
        config = load_config(cfg_file)
        model = build_model(config).to(device)
        train_loader, val_loader = build_dataloaders(config)
        
        optimizer_cfg = config['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg.get('weight_decay', 0.0))
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(config['trainer']['max_epochs']):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            val_loss = validate(model, criterion, val_loader, device)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save checkpoint
                ckpt_dir = Path(config['checkpoint']['dirpath'])
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"{cfg_file.stem}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping']['patience']:
                    print("Early stopping triggered")
                    break
        
        print(f"Training for {cfg_file.name} finished. Best val_loss={best_val_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory with YAML configs")
    args = parser.parse_args()
    main(args.config_dir)
