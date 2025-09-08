import datasets
import torch
from pathlib import Path
import pickle
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import gc


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
        text = re.sub(r"'s", " 's", text)
        text = re.sub(r"'t", " 't", text)
        text = re.sub(r"'re", " 're", text)
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"'ll", " 'll", text)
        text = re.sub(r"'d", " 'd", text)
        text = re.sub(r"'m", " 'm", text)
        
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
        most_common = self.word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, count in most_common:
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Final vocabulary size: {len(self.word2idx)} words")
        print(f"UNK token will be used for {sum(1 for c in self.word_counts.values() if c < self.min_freq)} rare words")
    
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


def create_sequences(token_ids: List[int], sequence_length: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create input/target sequences for language modeling
    Classical sliding window approach with stride=1
    """
    if len(token_ids) < sequence_length + 1:
        return [], []
    
    inputs = []
    targets = []
    
    # Sliding window with stride=1 (classical approach)
    for i in range(len(token_ids) - sequence_length):
        input_seq = token_ids[i:i + sequence_length]
        target_seq = token_ids[i + 1:i + sequence_length + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return inputs, targets


def process_split_classical(dataset_split, tokenizer: WikiTextTokenizer, 
                           sequence_length: int = 35) -> Dict[str, List]:
    """
    Process a dataset split using classical approach:
    1. Clean each article
    2. Tokenize each article separately
    3. Add EOS between articles
    4. Create sequences with sliding window (stride=1)
    """
    print(f"Processing {len(dataset_split)} articles...")
    
    # Concatenate all articles with EOS tokens between them
    all_tokens = []
    processed_articles = 0
    
    for example in dataset_split:
        text = example['text']
        
        # Clean the text
        cleaned_text = clean_wikitext(text)
        
        if not cleaned_text.strip():
            continue
        
        # Tokenize
        tokens = tokenizer.encode(cleaned_text)
        
        if tokens:  # Only add non-empty articles
            all_tokens.extend(tokens)
            # Add EOS token between articles (classical approach)
            all_tokens.append(tokenizer.word2idx[tokenizer.eos_token])
            processed_articles += 1
        
        if processed_articles % 1000 == 0:
            print(f"  Processed {processed_articles} articles, {len(all_tokens)} tokens so far...")
    
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Processed articles: {processed_articles}")
    
    # Create sequences
    print(f"Creating sequences with length {sequence_length}...")
    inputs, targets = create_sequences(all_tokens, sequence_length)
    
    print(f"Created {len(inputs)} sequences")
    
    return {
        'input_ids': inputs,
        'labels': targets,  # Use 'labels' for compatibility with transformers
        'length': [sequence_length] * len(inputs)  # All sequences have same length
    }


def preprocess_wikitext103_classical(data_path: str, output_path: str, 
                                   tokenizer_path: str, sequence_length: int = 35,
                                   vocab_size: int = 50000, min_freq: int = 1) -> None:
    """
    Complete classical preprocessing pipeline for WikiText-103
    
    Args:
        data_path: Path to raw WikiText-103 dataset (from HuggingFace)
        output_path: Where to save processed dataset
        tokenizer_path: Where to save/load tokenizer
        sequence_length: Length of sequences for language modeling
        vocab_size: Vocabulary size for tokenizer
        min_freq: Minimum frequency for words to be included in vocab
    """
    
    print("="*60)
    print("Classical WikiText-103 Preprocessing for Language Modeling")
    print("="*60)
    
    # Load raw dataset
    print(f"\n📥 Loading dataset from {data_path}...")
    if Path(data_path).exists():
        # Load from local path
        raw_dataset = datasets.load_from_disk(data_path)
    else:
        # Download from HuggingFace
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
        
        # Extract all training texts for vocabulary building
        train_texts = []
        print("Extracting training texts for vocabulary building...")
        for example in raw_dataset['train']:
            text = clean_wikitext(example['text'])
            if text.strip():
                train_texts.append(text)
        
        print(f"Extracted {len(train_texts)} training texts")
        
        # Build vocabulary
        tokenizer.build_vocab(train_texts)
        
        # Save tokenizer
        tokenizer.save(tokenizer_path)
        
        # Clean up
        del train_texts
        gc.collect()
    
    # Process each split
    processed_dataset = {}
    
    for split_name in ['train', 'validation', 'test']:
        print(f"\n🔄 Processing {split_name} split...")
        
        split_data = process_split_classical(
            raw_dataset[split_name], 
            tokenizer, 
            sequence_length=sequence_length
        )
        
        # Create HuggingFace dataset
        processed_dataset[split_name] = datasets.Dataset.from_dict(split_data)
        
        print(f"✅ {split_name} split processed:")
        print(f"   Sequences: {len(processed_dataset[split_name])}")
        print(f"   Tokens per sequence: {sequence_length}")
        
        # Clear memory
        del split_data
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
    input_ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
    labels = torch.tensor(batch['labels'], dtype=torch.long, device=device)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': torch.ones_like(input_ids)  # All tokens are real (no padding)
    }


if __name__ == "__main__":
    # Configuration - these are classical settings for WikiText-103
    DATA_PATH = "wikitext-103-raw"  # Will download if doesn't exist
    OUTPUT_PATH = "wikitext-103-processed"
    TOKENIZER_PATH = "wikitext103_tokenizer.pkl"
    
    # Classical WikiText-103 settings from the literature
    SEQUENCE_LENGTH = 35    # Classical setting from early papers
    VOCAB_SIZE = 50000      # Standard vocabulary size
    MIN_FREQ = 1            # Include all words (classical approach)
    
    print("Configuration:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Minimum Frequency: {MIN_FREQ}")
    
    # Run preprocessing
    dataset, tokenizer = preprocess_wikitext103_classical(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH, 
        tokenizer_path=TOKENIZER_PATH,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        min_freq=MIN_FREQ
    )
    
    # Example usage
    print(f"\n🔍 Example usage:")
    train_dataset = dataset['train']
    sample = train_dataset[0]
    
    print(f"Input sequence: {sample['input_ids']}")
    print(f"Target sequence: {sample['labels']}")
    print(f"Decoded input: {tokenizer.decode(sample['input_ids'])}")
    print(f"Decoded target: {tokenizer.decode(sample['labels'])}")