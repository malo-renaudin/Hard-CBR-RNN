import datasets
from pathlib import Path
from cbr_lightning.word_tok import WordTokenizer  # import your tokenizer class
import numpy as np


def preprocess_and_save(data_path, tokenizer, out_path, sequence_length=35, stride=35):
    raw = datasets.load_from_disk(data_path)

    tokenized_splits = {}
    for split in ["train", "validation", "test"]:
        texts = [t for t in raw[split]["text"] if t.strip()]
        token_ids = [tok for text in texts for tok in tokenizer.encode(text)]

        # Slice into sequences
        inputs, targets = [], []
        for i in range(0, len(token_ids) - sequence_length - 1, stride):
            seq = token_ids[i:i+sequence_length+1]
            inputs.append(seq[:-1])
            targets.append(seq[1:])

        tokenized_splits[split] = {
            "input_ids": inputs,
            "target_ids": targets,
        }

    # Save as Hugging Face DatasetDict
    ds = datasets.DatasetDict({
        split: datasets.Dataset.from_dict(tokenized_splits[split])
        for split in tokenized_splits
    })
    ds.save_to_disk(out_path)
    print(f"✅ Tokenized dataset saved at {out_path}")


if __name__ == "__main__":
    data_path = "cbr_lightning/wikitext-103-local"     # raw HF dataset
    out_path = "cbr_lightning/wikitext-103-tokenized"  # tokenized save path
    tokenizer_path = "./tokenizer.pkl"

    # Build/load tokenizer
    tokenizer = WordTokenizer(vocab_size=50000, min_freq=2)
    raw = datasets.load_from_disk(data_path)
    train_texts = [t for t in raw['train']['text'] if t.strip()]
    tokenizer.build_vocab(train_texts)
    tokenizer.save(tokenizer_path)

    # Preprocess & save
    preprocess_and_save(
        data_path=data_path,
        tokenizer=tokenizer,
        out_path=out_path,
        sequence_length=128,
        stride=64
    )
