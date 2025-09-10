import torch
from torch.utils.data import DataLoader

# --- import your code ---
from train2 import UniversalDataModule
from model_lightning import CBR_RNN, Transformer, LSTM
from word_tok import WordTokenizer  # adjust if tokenizer is elsewhere

import os

# ==== CONFIG ====
DATASET_PATH = "wikitext-103-tokenized"  # path where you saved datasets
TOKENIZER_PATH = "tokenizer.pkl"      # path to saved tokenizer
BATCH_SIZE = 4
SEQ_LEN = 128


def test_dataset_loading():
    dm = UniversalDataModule(dataset_path=DATASET_PATH,
                             tokenizer_path=TOKENIZER_PATH,
                             batch_size=BATCH_SIZE,
                             max_length=SEQ_LEN)
    dm.setup()

    assert dm.vocab_size > 0, "Tokenizer vocab is empty!"
    assert len(dm.dataset['train']) > 0, "Train dataset is empty!"
    assert len(dm.dataset['val']) > 0, "Validation dataset is empty!"
    assert len(dm.dataset['test']) > 0, "Test dataset is empty!"
    print("✅ Dataset loading works.")


def test_batch_shapes():
    dm = UniversalDataModule(dataset_path=DATASET_PATH,
                             tokenizer_path=TOKENIZER_PATH,
                             batch_size=BATCH_SIZE,
                             max_length=SEQ_LEN)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    inputs, targets = batch
    assert inputs.shape == (
        BATCH_SIZE, SEQ_LEN), f"Unexpected input shape {inputs.shape}"
    assert targets.shape == (
        BATCH_SIZE, SEQ_LEN), f"Unexpected target shape {targets.shape}"
    assert inputs.dtype == torch.long, "Inputs must be torch.long"
    assert targets.dtype == torch.long, "Targets must be torch.long"
    print("✅ Batch shapes and types are correct.")


def test_models_forward():
    dm = UniversalDataModule(dataset_path=DATASET_PATH,
                             tokenizer_path=TOKENIZER_PATH,
                             batch_size=BATCH_SIZE,
                             max_length=SEQ_LEN)
    dm.setup()
    inputs, targets = next(iter(dm.train_dataloader()))

    vocab_size = dm.vocab_size
    ntoken, ninp, nhid = vocab_size, 64, 128
    nheads, nlayers = 2, 2

    models = {
        "CBR_RNN": CBR_RNN(ntoken, ninp, nhid, nheads, SEQ_LEN,
                           compressed_dim=32, dropout=0.1,
                           learning_rate=1e-3, criterion="cross_entropy",
                           optimizer_type="adam"),
        "Transformer": Transformer(ntoken, ninp, nheads, nhid, nlayers,
                                   max_seq_len=SEQ_LEN,
                                   dropout=0.1, learning_rate=1e-3,
                                   criterion="cross_entropy",
                                   optimizer_type="adam"),
        "LSTM": LSTM(ntoken, ninp, nhid, nlayers,
                     dropout=0.1, learning_rate=1e-3,
                     criterion="cross_entropy",
                     optimizer_type="adam"),
    }

    for name, model in models.items():
        out = model(inputs, targets)  # Lightning model forward
        assert out is not None, f"{name} forward returned None"
        print(f"✅ {name} forward pass works.")

    print("✅ All models forward correctly.")


def test_loss_computation():
    dm = UniversalDataModule(dataset_path=DATASET_PATH,
                             tokenizer_path=TOKENIZER_PATH,
                             batch_size=BATCH_SIZE,
                             max_length=SEQ_LEN)
    dm.setup()
    inputs, targets = next(iter(dm.train_dataloader()))

    vocab_size = dm.vocab_size
    model = LSTM(ntoken=vocab_size, ninp=64, nhid=128, nlayers=2,
                 dropout=0.1, learning_rate=1e-3,
                 criterion="cross_entropy", optimizer_type="adam")

    loss = model.training_step((inputs, targets), 0)
    assert torch.is_tensor(loss), "Loss must be a tensor"
    assert loss.item() > 0, "Loss should be positive"
    print("✅ Loss computation works.")


if __name__ == "__main__":
    print("🚀 Running pipeline tests...")
    test_dataset_loading()
    test_batch_shapes()
    test_models_forward()
    test_loss_computation()
    print("🎉 All tests passed successfully.")
