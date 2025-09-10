import torch
from tqdm import trange

# Import model factory and datamodule directly
from train2 import UniversalDataModule, ModelFactory


def get_one_batch(data_module):
    dl = data_module.train_dataloader()
    batch = next(iter(dl))
    if isinstance(batch, dict):  # Original code
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda() if torch.cuda.is_available() else v
        return batch
    elif isinstance(batch, tuple) and len(batch) == 2:  # Handle tuple case
        inputs, targets = batch
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        targets = targets.cuda() if torch.cuda.is_available() else targets
        return {'input_ids': inputs, 'target_ids': targets}
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")


def run_overfit_all_models(steps=500):
    # Use default dataset/tokenizer info
    data_config = {
        'dataset_path': './wikitext-103-tokenized',
        'tokenizer_path': '../tokenizer.pkl',
        'batch_size': 8,          # Small batch for fast overfit
        'num_workers': 0,
        'max_length': 128         # Short sequence for debugging
        # 'vocab_size': 50000       # Will be auto-detected if possible
    }
    # Load data
    data_module = UniversalDataModule(**data_config)
    data_module.setup()
    batch = get_one_batch(data_module)
    print("Sample batch keys:", batch.keys())

    # Prepare input/target
    if 'input_ids' in batch:
        input_ids = batch['input_ids']
    elif 'inputs' in batch:
        input_ids = batch['inputs']
    else:
        raise KeyError("Batch does not contain 'input_ids' or 'inputs'")
    if 'target_ids' in batch:
        target_ids = batch['target_ids']
    elif 'targets' in batch:
        target_ids = batch['targets']
    else:
        raise KeyError("Batch does not contain 'target_ids' or 'targets'")
    if target_ids.ndim > 1:
        target_ids = target_ids.view(-1)

    # Model types and their minimal configs
    model_types = {
        "CBR_RNN": {
            'ninp': 32, 'nhid': 64, 'nheads': 2, 'compressed_dim': 8, 'dropout': 0.1,
            'temperature': 1.0, 'gumbel_softmax': False
        },
        "Transformer": {
            'd_model': 32, 'n_heads': 2, 'n_layers': 2, 'd_ff': 64, 'dropout': 0.1,
            'temperature': 1.0, 'gumbel_softmax': False
        },
        "LSTM": {
            'embedding_dim': 32, 'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1
        }
    }

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_type, model_config in model_types.items():
        print(f"\n--- Overfitting {model_type} ---")
        model = ModelFactory.create_model(
            model_type, model_config, data_module.vocab_size)
        model.train()
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if model_type == "CBR_RNN":
            input_ids = input_ids.transpose(0, 1)

        # Forward pass: adjust for model signature
        for step in trange(steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            if logits.ndim == 3:
                logits = logits.view(-1, logits.size(-1))
            loss = criterion(logits, target_ids)
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == steps - 1:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        print(f"Final loss for {model_type}: {loss.item():.4f}")


if __name__ == "__main__":
    run_overfit_all_models()
