from itertools import product
import json
from pathlib import Path

def create_configs(save_dir=None):
    params = {
        'all': {'model': ['CBR_RNN', 'Transformer'], 'nhid': [256, 512, 1024], 
                'nlayers': [1, 2, 4], 'dropout': [0.1, 0.3, 0.5]},
        'attn': {'nheads': [1, 4, 8], 'temp_decay': [0.95, 0.99, 0.999], 
                 'end_temp': [0.1, 0.5, 1.0]}
    }
    
    attn_models = {'CBR_RNN', 'Transformer'}
    all_keys, all_vals = zip(*params['all'].items())
    attn_keys, attn_vals = zip(*params['attn'].items())
    
    configs = [
        {**dict(zip(all_keys, base)), **dict(zip(attn_keys, attn))}
        for base in product(*all_vals)
        for attn in (product(*attn_vals) if (model := dict(zip(all_keys, base))['model']) in attn_models else [{}])
        if model not in attn_models or dict(zip(all_keys, base))['nhid'] % dict(zip(attn_keys, attn)).get('nheads', 1) == 0
    ]
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / 'configs.json', 'w') as f:
            json.dump(configs, f, indent=2)
    
    return configs

# Usage
configs = create_configs(save_dir='./experiments/configs')
print(f"Generated {len(configs)} configurations")