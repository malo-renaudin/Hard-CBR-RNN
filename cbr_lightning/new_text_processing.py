import datasets
import torch
from pathlib import Path
import pickle
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Generator
import gc
import numpy as np


class WikiTextTokenizer:
    """
    Classical word-level tokenizer for WikiText-103
    Follows the original preprocessing approach from the dataset paper
    """

    def __init__(self, vocab_size: int = 50000, min_freq: int = 1,
                 unk_token: str = "<unk>", eos_token: str = "<eos>"):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.eos_token = eos_token

        # Special tokens
        self.special_tokens = [unk_token, eos_token]

        # Vocabularies
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Classical tokenization: split on whitespace and basic punctuation
        This matches the original WikiText preprocessing
        """
        # Handle common contractions and punctuation
        # text = re.sub(r"'s", " 's", text)
        # text = re.sub(r"'t", " 't", text)
        # text = re.sub(r"'re", " 're", text)
        # text = re.sub(r"'ve", " 've", text)
        # text = re.sub(r"'ll", " 'll", text)
        # text = re.sub(r"'d", " 'd", text)
        # text = re.sub(r"'m", " 'm", text)

        # Split on punctuation but keep it
        text = re.sub(r"([.!?,:;])", r" \1 ", text)
        text = re.sub(r"([()])", r" \1 ", text)
        text = re.sub(r"([\[\]])", r" \1 ", text)
        text = re.sub(r"([{}])", r" \1 ", text)
        text = re.sub(r'([""])', r' \1 ', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip().split()

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        print("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            if not text.strip():
                continue

            words = self._tokenize_text(text)
            self.word_counts.update(words)

        print(f"Found {len(self.word_counts)} unique words")

        # Create vocabulary with special tokens first
        self.word2idx = {}
        self.idx2word = {}

        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

        # Add most frequent words
        most_common = self.word_counts.most_common(
            self.vocab_size - len(self.special_tokens))

        for word, count in most_common:
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Final vocabulary size: {len(self.word2idx)} words")
        print(
            f"UNK token will be used for {sum(1 for c in self.word_counts.values() if c < self.min_freq)} rare words")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        if not text.strip():
            return []

        words = self._tokenize_text(text)
        token_ids = []

        for word in words:
            if word in self.word2idx:
                token_ids.append(self.word2idx[word])
            else:
                token_ids.append(self.word2idx[self.unk_token])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.idx2word:
                words.append(self.idx2word[token_id])
            else:
                words.append(self.unk_token)

        return " ".join(words)

    def save(self, path: str) -> None:
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'unk_token': self.unk_token,
            'eos_token': self.eos_token,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts)
        }

        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str) -> None:
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)

        self.vocab_size = tokenizer_data['vocab_size']
        self.min_freq = tokenizer_data['min_freq']
        self.unk_token = tokenizer_data['unk_token']
        self.eos_token = tokenizer_data['eos_token']
        self.word2idx = tokenizer_data['word2idx']
        self.idx2word = tokenizer_data['idx2word']
        self.word_counts = Counter(tokenizer_data['word_counts'])

        print(f"Tokenizer loaded from {path}")
        print(f"Vocabulary size: {len(self.word2idx)} words")


def clean_wikitext(text: str) -> str:
    """
    Clean WikiText-103 specific formatting
    This follows the classical approach used in language modeling papers
    """
    if not text.strip():
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle section headers - convert to regular text
        if line.startswith('=') and line.endswith('='):
            # Extract section title
            section_title = line.strip('= ').strip()
            if section_title:
                cleaned_lines.append(section_title)
        else:
            cleaned_lines.append(line)

    # Join with spaces (classical approach doesn't preserve line breaks)
    return ' '.join(cleaned_lines)


def create_sequences_generator(token_ids: List[int], sequence_length: int,
                               chunk_size: int = 10000) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Memory-efficient sequence generator using chunks
    Yields (input_seq, target_seq) pairs
    """
    if len(token_ids) < sequence_length + 1:
        return

    # Process in chunks to avoid memory issues
    for start_idx in range(0, len(token_ids) - sequence_length, chunk_size):
        end_idx = min(start_idx + chunk_size, len(token_ids) - sequence_length)

        # Generate sequences for this chunk
        for i in range(start_idx, end_idx):
            input_seq = token_ids[i:i + sequence_length]
            target_seq = token_ids[i + 1:i + sequence_length + 1]
            yield input_seq, target_seq


def process_articles_streaming(dataset_split, tokenizer: WikiTextTokenizer,
                               max_articles: Optional[int] = None) -> Generator[int, None, None]:
    """
    Stream and tokenize articles one by one to avoid memory buildup
    Yields individual tokens
    """
    processed_articles = 0

    for example in dataset_split:
        if max_articles and processed_articles >= max_articles:
            break

        text = example['text']

        # Clean the text
        cleaned_text = clean_wikitext(text)

        if not cleaned_text.strip():
            continue

        # Tokenize
        tokens = tokenizer.encode(cleaned_text)

        if tokens:  # Only process non-empty articles
            # Yield all tokens from this article
            for token in tokens:
                yield token

            # Add EOS token between articles
            yield tokenizer.word2idx[tokenizer.eos_token]
            processed_articles += 1

        if processed_articles % 1000 == 0:
            print(f"  Processed {processed_articles} articles...")

    print(f"Total processed articles: {processed_articles}")


def process_split_memory_efficient(dataset_split, tokenizer: WikiTextTokenizer,
                                   sequence_length: int = 35,
                                   batch_size: int = 1000,
                                   max_articles: Optional[int] = None) -> datasets.Dataset:
    """
    Memory-efficient processing of dataset split
    """
    print(f"Processing {len(dataset_split)} articles (memory-efficient)...")

    # First pass: collect all tokens in chunks
    all_sequences_input = []
    all_sequences_target = []
    current_batch_input = []
    current_batch_target = []

    # Stream tokens and create sequences in batches
    print("Streaming articles and creating sequences...")
    token_buffer = []
    buffer_size = sequence_length * 1000  # Buffer to create sequences efficiently

    total_sequences = 0

    for token in process_articles_streaming(dataset_split, tokenizer, max_articles):
        token_buffer.append(token)

        # When buffer is full, create sequences and clear buffer
        if len(token_buffer) >= buffer_size:
            # Create sequences from buffer
            for input_seq, target_seq in create_sequences_generator(token_buffer, sequence_length):
                current_batch_input.append(input_seq)
                current_batch_target.append(target_seq)

                # When batch is full, add to main list and clear
                if len(current_batch_input) >= batch_size:
                    all_sequences_input.extend(current_batch_input)
                    all_sequences_target.extend(current_batch_target)
                    total_sequences += len(current_batch_input)

                    print(f"  Created {total_sequences} sequences so far...")

                    current_batch_input = []
                    current_batch_target = []

                    # Cleanup to avoid memory buildup
                    gc.collect()

            # Keep overlapping tokens for continuity
            overlap_size = sequence_length
            token_buffer = token_buffer[-overlap_size:]

    # Process remaining tokens in buffer
    if len(token_buffer) >= sequence_length + 1:
        for input_seq, target_seq in create_sequences_generator(token_buffer, sequence_length):
            current_batch_input.append(input_seq)
            current_batch_target.append(target_seq)

    # Add final batch
    if current_batch_input:
        all_sequences_input.extend(current_batch_input)
        all_sequences_target.extend(current_batch_target)
        total_sequences += len(current_batch_input)

    print(f"Created {total_sequences} total sequences")

    # Create dataset
    dataset_dict = {
        'input_ids': all_sequences_input,
        'target_ids': all_sequences_target,
        'length': [sequence_length] * len(all_sequences_input)
    }

    return datasets.Dataset.from_dict(dataset_dict)


def preprocess_wikitext103_memory_efficient(data_path: str, output_path: str,
                                            tokenizer_path: str, sequence_length: int = 35,
                                            vocab_size: int = 50000, min_freq: int = 1,
                                            max_train_articles: Optional[int] = None) -> None:
    """
    Memory-efficient preprocessing pipeline for WikiText-103

    Args:
        data_path: Path to raw WikiText-103 dataset (from HuggingFace)
        output_path: Where to save processed dataset
        tokenizer_path: Where to save/load tokenizer
        sequence_length: Length of sequences for language modeling
        vocab_size: Vocabulary size for tokenizer
        min_freq: Minimum frequency for words to be included in vocab
        max_train_articles: Limit training articles for testing (None = all)
    """

    print("="*60)
    print("Memory-Efficient WikiText-103 Preprocessing")
    print("="*60)

    # Load raw dataset
    print(f"\n📥 Loading dataset from {data_path}...")
    if Path(data_path).exists():
        raw_dataset = datasets.load_from_disk(data_path)
    else:
        print("Dataset not found locally, downloading from HuggingFace...")
        raw_dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")
        raw_dataset.save_to_disk(data_path)

    print(f"Dataset splits: {list(raw_dataset.keys())}")
    for split in raw_dataset.keys():
        print(f"  {split}: {len(raw_dataset[split])} examples")

    # Initialize or load tokenizer
    tokenizer_path_obj = Path(tokenizer_path)
    if tokenizer_path_obj.exists():
        print(f"\n📚 Loading existing tokenizer from {tokenizer_path}...")
        tokenizer = WikiTextTokenizer(vocab_size=vocab_size, min_freq=min_freq)
        tokenizer.load(tokenizer_path)
    else:
        print(f"\n📚 Building new tokenizer...")
        tokenizer = WikiTextTokenizer(vocab_size=vocab_size, min_freq=min_freq)

        # Extract training texts for vocabulary building (in chunks to save memory)
        train_texts = []
        print("Extracting training texts for vocabulary building...")

        # Limit for vocab building
        max_vocab_articles = min(10000, len(raw_dataset['train']))
        for i, example in enumerate(raw_dataset['train']):
            if i >= max_vocab_articles:
                break

            text = clean_wikitext(example['text'])
            if text.strip():
                train_texts.append(text)

            if i % 1000 == 0:
                print(f"  Extracted {len(train_texts)} texts so far...")

        print(f"Extracted {len(train_texts)} training texts for vocabulary")

        # Build vocabulary
        tokenizer.build_vocab(train_texts)
        tokenizer.save(tokenizer_path)

        # Clean up
        del train_texts
        gc.collect()

    # Process each split with memory efficiency
    processed_dataset = {}

    split_limits = {
        'train': max_train_articles,  # Can be limited for testing
        'validation': None,  # Process all validation
        'test': None  # Process all test
    }

    for split_name in ['train', 'validation', 'test']:
        print(f"\n🔄 Processing {split_name} split...")

        processed_split = process_split_memory_efficient(
            raw_dataset[split_name],
            tokenizer,
            sequence_length=sequence_length,
            max_articles=split_limits[split_name]
        )

        processed_dataset[split_name] = processed_split

        print(f"✅ {split_name} split processed:")
        print(f"   Sequences: {len(processed_dataset[split_name])}")
        print(f"   Tokens per sequence: {sequence_length}")

        # Clear memory
        gc.collect()

    # Create DatasetDict and save
    print(f"\n💾 Saving processed dataset to {output_path}...")
    final_dataset = datasets.DatasetDict(processed_dataset)
    final_dataset.save_to_disk(output_path)

    print(f"\n✅ Processing complete!")
    print(f"📁 Dataset saved to: {output_path}")
    print(f"📁 Tokenizer saved to: {tokenizer_path}")

    # Print final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"   Sequence length: {sequence_length}")
    for split in final_dataset.keys():
        print(f"   {split}: {len(final_dataset[split]):,} sequences")

    return final_dataset, tokenizer


def create_dataloader_batch(batch, device='cpu'):
    """
    Convert batch to PyTorch tensors for training
    Classical language modeling setup
    """
    input_ids = torch.tensor(
        batch['input_ids'], dtype=torch.long, device=device)
    labels = torch.tensor(batch['labels'], dtype=torch.long, device=device)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': torch.ones_like(input_ids)
    }


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "wikitext-103-raw"
    OUTPUT_PATH = "wikitext-103-processed"
    TOKENIZER_PATH = "wikitext103_tokenizer.pkl"

    # Memory-efficient settings
    SEQUENCE_LENGTH = 35
    VOCAB_SIZE = 50000
    MIN_FREQ = 1

    # For testing - limit training articles (set to None for full dataset)
    MAX_TRAIN_ARTICLES = 5000  # Set to None for full dataset

    print("Memory-Efficient Configuration:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Max Training Articles: {MAX_TRAIN_ARTICLES}")

    # Run preprocessing
    dataset, tokenizer = preprocess_wikitext103_memory_efficient(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        tokenizer_path=TOKENIZER_PATH,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        min_freq=MIN_FREQ,
        max_train_articles=MAX_TRAIN_ARTICLES
    )

    # Example usage
    print(f"\n🔍 Example usage:")
    train_dataset = dataset['train']
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        # Show first 10 tokens
        print(f"Input sequence: {sample['input_ids'][:10]}...")
        print(f"Target sequence: {sample['labels'][:10]}...")
        print(
            f"Decoded input: {tokenizer.decode(sample['input_ids'][:10])}...")
