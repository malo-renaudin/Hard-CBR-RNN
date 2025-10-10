"""Data utilities for WikiText processing"""
import torch
from torch.utils.data import Dataset
from collections import Counter
import json
import re
from nltk.tokenize import sent_tokenize


def normalize_ptb(word):
    """Normalize Penn Treebank special tokens"""
    replacements = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LCB-': '{',
        '-RCB-': '}',
        '-LSB-': '[',
        '-RSB-': ']',
        '``': '"',
        "''": '"',
        '--': '—',
    }
    for old, new in replacements.items():
        word = word.replace(old, new)
    return word


def normalize_numbers(word):
    """Replace clear numeric patterns with <num> token"""
    # Pure integers: 123, 4567
    if word.isdigit():
        return '<num>'
    
    # Numbers with commas: 1,234
    if re.match(r'^\d{1,3}(,\d{3})+$', word):
        return '<num>'
    
    # Decimals: 3.14, .5, 123.456
    if re.match(r'^\d*\.\d+$', word) or re.match(r'^\d+\.\d*$', word):
        return '<num>'
    
    # Keep everything else as-is (preserves hyphenated words, abbreviations, etc.)
    return word


def clean_wikitext(text):
    """Clean WikiText-103 specific patterns"""
    # Remove @-@ (Penn Treebank hyphen marker)
    text = text.replace(' @-@ ', '-')
    text = text.replace('@-@', '-')
    
    # Remove @,@ (comma marker)
    text = text.replace(' @,@ ', ', ')
    text = text.replace('@,@', ',')
    
    # Remove @.@ (period marker)
    text = text.replace(' @.@ ', '. ')
    text = text.replace('@.@', '.')
    
    # Better section marker handling: extract the title text
    # "= Title =" → "Title" or "== Section ==" → "Section"
    text = re.sub(r'=+\s*(.+?)\s*=+', r'\1', text)
    
    # Also handle lines that are just "=" markers
    text = re.sub(r'^=+\s*$', '', text, flags=re.MULTILINE)
    
    return text


class WordTokenizer:
    """Word-level tokenizer for WikiText with normalization and lowercasing"""
    def __init__(self, list_of_texts, vocab_size=50000):
        
        self.special_tokens = {
            '<unk>': 0,
            '<boa>': 1,  # beginning of article
            '<eos>': 2,  # end of sentence
            '<eoa>': 3,  # end of article
            '<num>': 4,  # number token
        }
        
        # Collect and normalize tokens
        tokens = []
        for text in list_of_texts:
            if text.strip():
                # Clean WikiText markup
                text = clean_wikitext(text)
                # Lowercase everything
                text = text.lower()
                words = text.split()
                
                # Normalize each word
                normalized = [
                    normalize_numbers(normalize_ptb(word)) 
                    for word in words
                ]
                tokens.extend(normalized)
        
        counter = Counter(tokens)
        num_special = len(self.special_tokens)
        most_common = counter.most_common(vocab_size - num_special)
        
        self.itos = list(self.special_tokens.keys()) + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        
        # Store special token IDs
        self.unk_id = self.special_tokens['<unk>']
        self.boa_id = self.special_tokens['<boa>']
        self.eos_id = self.special_tokens['<eos>']
        self.eoa_id = self.special_tokens['<eoa>']
        self.num_id = self.special_tokens['<num>']

    def encode(self, text):
        """Encode text with sentence and article boundaries"""
        if not text.strip():
            return []
        
        tokens = []
        
        # Add beginning of article token
        tokens.append(self.boa_id)
        
        # Lowercase the text
        text = text.lower()
        
        # Split into sentences using NLTK
        sentences = sent_tokenize(text)
        
        for sent in sentences:
            words = sent.split()
            # Normalize during encoding
            for word in words:
                word = normalize_ptb(word)
                word = normalize_numbers(word)
                tokens.append(self.stoi.get(word, self.unk_id))
            # Add end of sentence token
            tokens.append(self.eos_id)
        
        # Add end of article token
        tokens.append(self.eoa_id)
        
        return tokens

    def decode(self, ids):
        """Decode token IDs back to text"""
        return " ".join([self.itos[i] for i in ids])

    def save(self, path):
        """Save tokenizer to file"""
        with open(path, 'w') as f:
            json.dump({
                'itos': self.itos,
                'stoi': self.stoi,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls.__new__(cls)
        tokenizer.itos = data['itos']
        tokenizer.stoi = data['stoi']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.special_tokens = data['special_tokens']
        
        # Restore special token IDs
        tokenizer.unk_id = tokenizer.special_tokens['<unk>']
        tokenizer.boa_id = tokenizer.special_tokens['<boa>']
        tokenizer.eos_id = tokenizer.special_tokens['<eos>']
        tokenizer.eoa_id = tokenizer.special_tokens['<eoa>']
        tokenizer.num_id = tokenizer.special_tokens['<num>']
        
        return tokenizer


class WikiTextDataset(Dataset):
    """Dataset for WikiText that produces sequences with boundary tokens"""
    def __init__(self, dataset, tokenizer, seq_len=64):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Process articles: in WikiText, articles are separated by empty lines
        all_tokens = []
        current_article = []
        
        for line in dataset["text"]:
            line_stripped = line.strip()
            
            if line_stripped:  # Non-empty line
                current_article.append(line_stripped)
            else:  # Empty line - marks article boundary
                if current_article:
                    # Join all lines of the article and clean
                    article_text = " ".join(current_article)
                    article_text = clean_wikitext(article_text)
                    tokens = tokenizer.encode(article_text)
                    all_tokens.extend(tokens)
                    current_article = []
        
        # Don't forget the last article if file doesn't end with empty line
        if current_article:
            article_text = " ".join(current_article)
            article_text = clean_wikitext(article_text)
            tokens = tokenizer.encode(article_text)
            all_tokens.extend(tokens)
        
        self.data = torch.tensor(all_tokens, dtype=torch.long)
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.data[i:i+self.seq_len]
        target = self.data[i+1:i+self.seq_len+1]
        return seq, target


class PreprocessedWikiTextDataset(Dataset):
    """Dataset loaded from preprocessed tokenized data"""
    def __init__(self, data, seq_len=64):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        i = idx * self.seq_len
        seq = self.data[i:i+self.seq_len]
        target = self.data[i+1:i+self.seq_len+1]
        return seq, target