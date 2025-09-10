#!/usr/bin/env python3
"""
Generate YAML configs for all models
"""

import yaml
from pathlib import Path

# --- Output directory for configs ---
config_dir = Path("configs")
config_dir.mkdir(exist_ok=True)

# --- Define models and default parameters ---
models = {
    "CBR_RNN": {
        "type": "CBR_RNN",
        "config": {
            "ntoken": 50000,
            "ninp": 512,
            "nhid": 512,
            "nheads": 8,
            "seq_len": 128,
            "compressed_dim": 64,
            "dropout": 0.1,
        }
    },
    "Transformer": {
        "type": "Transformer",
        "config": {
            "ntoken": 50000,
            "d_model":256,
            "d_ff": 256,
            "nheads": 8,
            "nlayers": 6,
            "seq_len": 128,
            "dropout": 0.1,
        }
    },
    "LSTM": {
        "type": "LSTM",
        "config": {
            "ntoken": 50000,
            "ninp": 512,
            "nhid": 512,
            "nlayers": 2,
            "seq_len": 128,
            "dropout": 0.1,
        }
    },
}

# --- Default training/data/optimizer configs ---
common_cfg = {
    "data": {
        "dataset_path": "data/dataset.pkl",
        "batch_size": 512,
        "num_workers": 4,
        "max_length": 128
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0
    },
    "trainer": {
        "max_epochs": 5
    },
    "early_stopping": {
        "patience": 5
    },
    "checkpoint": {
        "dirpath": "checkpoints"
    }
}

# --- Generate YAML files ---
for model_name, model_cfg in models.items():
    cfg = {
        "model": model_cfg,
        **common_cfg
    }
    output_file = config_dir / f"{model_name}.yaml"
    with open(output_file, "w") as f:
        yaml.dump(cfg, f)
    print(f"Generated {output_file}")
