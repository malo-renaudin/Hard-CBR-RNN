"""Prepare shared data (tokenizer and datasets) once before running experiments"""
import pytorch_lightning as pl
import datasets
from pathlib import Path
from data_utils import WordTokenizer


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
    
    print(f"Tokenizer saved with vocab size: {tokenizer.vocab_size}")
    print(f"Saved to: {shared_dir / 'tokenizer.json'}")
    
    return tokenizer.vocab_size


if __name__ == "__main__":
    vocab_size = prepare_shared_data()
    print(f"\nData preparation complete!")
    print(f"Vocabulary size: {vocab_size}")