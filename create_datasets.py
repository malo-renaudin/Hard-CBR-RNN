from datasets import load_from_disk
from grid_search.data_utils import WordTokenizer, WikiTextDataset
import torch
import nltk

nltk.download('punkt_tab')

# Load your local WikiText-103-raw dataset
train_dataset = load_from_disk("wikitext-103-raw/train")
valid_dataset = load_from_disk("wikitext-103-raw/validation")
test_dataset = load_from_disk("wikitext-103-raw/test")

# Create tokenizer with special tokens
print("Building tokenizer...")
train_texts = train_dataset["text"]
tokenizer = WordTokenizer(train_texts, vocab_size=50000)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens}")

# Create datasets (much simpler now!)
print("Creating training dataset...")
train_data = WikiTextDataset(train_dataset, tokenizer, seq_len=64)

print("Creating validation dataset...")
valid_data = WikiTextDataset(valid_dataset, tokenizer, seq_len=64)

print("Creating test dataset...")
test_data = WikiTextDataset(test_dataset, tokenizer, seq_len=64)

print(f"Train dataset size: {len(train_data)} sequences")
print(f"Valid dataset size: {len(valid_data)} sequences")
print(f"Test dataset size: {len(test_data)} sequences")

# Save tokenizer
print("\nSaving tokenizer...")
tokenizer.save("tokenizer.json")

# Save datasets
print("Saving datasets...")
torch.save(train_data.data, 'train_data.pt')
torch.save(valid_data.data, 'valid_data.pt')
torch.save(test_data.data, 'test_data.pt')

print("All datasets saved!")

# Example: check a sample
sample_seq, sample_target = train_data[10]
print(f"\nSample sequence shape: {sample_seq.shape}")
print(f"Decoded: {tokenizer.decode(sample_seq[:20].tolist())}")