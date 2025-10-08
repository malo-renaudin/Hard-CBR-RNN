"""Experiment configurations for grid search"""


def get_all_configs():
    """Return all experiment configurations"""
    configs = [
        {
            'nhid': 512,
            'nheads': 1,
            'lr': 5e-4,
            'dropout': 0.5,
            'use_gumbel_softmax': True,
            'final_temp': 0.5,
            'temp_decay': 'exponential'
        },
        {
            'nhid': 512,
            'nheads': 1,
            'lr': 5e-4,
            'dropout': 0.5,
            'use_gumbel_softmax': True,
            'final_temp': 0.1,
            'temp_decay': 'linear'
        },
        {
            'nhid': 512,
            'nheads': 1,
            'lr': 5e-4,
            'dropout': 0.5,
            'use_gumbel_softmax': True,
            'final_temp': 0.5,
            'temp_decay': 'linear'
        },
        {
            'nhid': 512,
            'nheads': 1,
            'lr': 5e-4,
            'dropout': 0.5,
            'use_gumbel_softmax': True,
            'final_temp': 0.1,
            'temp_decay': 'cosine'
        },
        {
            'nhid': 512,
            'nheads': 1,
            'lr': 5e-4,
            'dropout': 0.5,
            'use_gumbel_softmax': True,
            'final_temp': 0.5,
            'temp_decay': 'cosine'
        }
    ]
    return configs