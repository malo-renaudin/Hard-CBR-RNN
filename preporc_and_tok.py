import datasets
from pathlib import Path
from cbr_lightning.word_tok import WordTokenizer
import numpy as np
import gc
from typing import Generator, List, Iterator


def chunked_text_generator(dataset_split, chunk_size: int = 1000) -> Generator[List[str], None, None]:
    """
    Generate chunks of text from dataset to avoid loading all at once
    """
    texts = []
    for item in dataset_split:
        if item["text"].strip():
            texts.append(item["text"])
            if len(texts) >= chunk_size:
                yield texts
                texts = []
    if texts:  # yield remaining texts
        yield texts


def tokenize_chunk_generator(text_chunks: List[str], tokenizer) -> Generator[int, None, None]:
    """
    Tokenize text chunks and yield individual tokens
    """
    for text in text_chunks:
        for token_id in tokenizer.encode(text):
            yield token_id


def sequence_generator(token_generator: Generator[int, None, None], 
                      sequence_length: int, 
                      stride: int) -> Generator[tuple, None, None]:
    """
    Generate input/target sequences from token stream without storing all tokens
    """
    # Buffer to hold tokens for sequence creation
    buffer = []
    
    for token in token_generator:
        buffer.append(token)
        
        # Once we have enough tokens, start yielding sequences
        while len(buffer) >= sequence_length + 1:
            seq = buffer[:sequence_length + 1]
            input_seq = seq[:-1]
            target_seq = seq[1:]
            
            yield input_seq, target_seq
            
            # Move buffer by stride
            if stride < len(buffer):
                buffer = buffer[stride:]
            else:
                buffer = []
                break


def build_vocab_identical(dataset_split, tokenizer):
    """
    Build vocabulary EXACTLY like the original script - load all texts at once
    This ensures identical tokenizer to the simple script
    """
    print("Building vocabulary identically to original script...")
    
    # Load all texts at once (same as original script)
    train_texts = [t for t in dataset_split["text"] if t.strip()]
    print(f"  Loaded {len(train_texts)} training texts")
    
    # Use the original build_vocab method
    tokenizer.build_vocab(train_texts)
    print(f"  Vocabulary built with {len(tokenizer.word2idx)} tokens")
    
    # Clean up
    del train_texts
    gc.collect()


def preprocess_split_chunked(dataset_split, tokenizer, sequence_length: int = 35, 
                           stride: int = 35, text_chunk_size: int = 1000,
                           batch_size: int = 10000) -> datasets.Dataset:
    """
    Process a single split in memory-efficient chunks
    """
    all_inputs = []
    all_targets = []
    current_batch_size = 0
    
    print(f"Processing split with {len(dataset_split)} examples...")
    
    # Process in chunks
    for chunk_idx, text_chunk in enumerate(chunked_text_generator(dataset_split, text_chunk_size)):
        print(f"  Processing text chunk {chunk_idx + 1} ({len(text_chunk)} texts)...")
        
        # Create token generator for this chunk
        token_gen = tokenize_chunk_generator(text_chunk, tokenizer)
        
        # Create sequence generator
        seq_gen = sequence_generator(token_gen, sequence_length, stride)
        
        # Collect sequences in batches
        chunk_inputs = []
        chunk_targets = []
        
        for input_seq, target_seq in seq_gen:
            chunk_inputs.append(input_seq)
            chunk_targets.append(target_seq)
            current_batch_size += 1
            
            # If batch is full, add to main lists and clear
            if current_batch_size >= batch_size:
                all_inputs.extend(chunk_inputs)
                all_targets.extend(chunk_targets)
                chunk_inputs = []
                chunk_targets = []
                current_batch_size = 0
                
                print(f"    Accumulated {len(all_inputs)} sequences so far...")
        
        # Add remaining sequences from this chunk
        if chunk_inputs:
            all_inputs.extend(chunk_inputs)
            all_targets.extend(chunk_targets)
        
        # Force garbage collection after each chunk
        del text_chunk, token_gen, seq_gen, chunk_inputs, chunk_targets
        gc.collect()
    
    print(f"  Total sequences created: {len(all_inputs)}")
    
    # Create dataset
    dataset_dict = {
        "input_ids": all_inputs,
        "target_ids": all_targets,
    }
    
    dataset = datasets.Dataset.from_dict(dataset_dict)
    
    # Clear memory
    del all_inputs, all_targets, dataset_dict
    gc.collect()
    
    return dataset


def load_tokenizer(tokenizer_path: str) -> WordTokenizer:
    """
    Load existing tokenizer from pickle file
    """
    tokenizer = WordTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def preprocess_and_save_efficient(data_path, tokenizer, out_path, 
                                sequence_length=35, stride=35, 
                                text_chunk_size=1000, batch_size=10000):
    """
    Memory-efficient preprocessing that handles large datasets
    Uses IDENTICAL tokenizer building as the original script
    """
    print(f"Loading raw dataset from {data_path}...")
    raw = datasets.load_from_disk(data_path)
    
    tokenized_splits = {}
    
    for split in ["train", "validation", "test"]:
        print(f"\n🔄 Processing {split} split...")
        
        # Process split efficiently
        dataset = preprocess_split_chunked(
            raw[split], 
            tokenizer, 
            sequence_length=sequence_length,
            stride=stride,
            text_chunk_size=text_chunk_size,
            batch_size=batch_size
        )
        
        tokenized_splits[split] = dataset
        
        print(f"✅ {split} split processed: {len(dataset)} sequences")
        
        # Force garbage collection between splits
        gc.collect()
    
    # Create DatasetDict
    print("\n📦 Creating DatasetDict...")
    ds = datasets.DatasetDict(tokenized_splits)
    
    # Save to disk
    print(f"💾 Saving to {out_path}...")
    ds.save_to_disk(out_path)
    
    print(f"✅ Tokenized dataset saved at {out_path}")
    
    # Print final statistics
    for split in ds:
        print(f"  {split}: {len(ds[split])} sequences")
    
    return ds


def estimate_memory_usage(data_path, sequence_length=128, stride=64):
    """
    Estimate memory usage and provide recommendations
    """
    print("🔍 Estimating memory requirements...")
    
    raw = datasets.load_from_disk(data_path)
    
    total_chars = 0
    total_sequences_estimate = 0
    
    for split in ["train", "validation", "test"]:
        split_chars = sum(len(text) for text in raw[split]["text"] if text.strip())
        total_chars += split_chars
        
        # Rough estimate: 1 token per 4-5 characters
        estimated_tokens = split_chars // 4
        estimated_sequences = max(0, (estimated_tokens - sequence_length) // stride)
        total_sequences_estimate += estimated_sequences
        
        print(f"  {split}: ~{split_chars:,} chars, ~{estimated_tokens:,} tokens, ~{estimated_sequences:,} sequences")
    
    # Memory estimates
    bytes_per_sequence = sequence_length * 4  # int32
    estimated_memory_gb = (total_sequences_estimate * bytes_per_sequence * 2) / (1024**3)  # *2 for input+target
    
    print(f"\n📊 Total estimated:")
    print(f"  Characters: {total_chars:,}")
    print(f"  Sequences: {total_sequences_estimate:,}")
    print(f"  Memory needed: ~{estimated_memory_gb:.2f} GB")
    
    # Recommendations
    if estimated_memory_gb > 16:
        print("\n⚠️  High memory usage detected! Recommendations:")
        print(f"  - Use text_chunk_size=500 (default: 1000)")
        print(f"  - Use batch_size=5000 (default: 10000)")
        print(f"  - Consider smaller sequence_length or larger stride")
    
    return total_sequences_estimate, estimated_memory_gb


if __name__ == "__main__":
    data_path = "cbr_lightning/wikitext-103-local"  # raw HF dataset
    out_path = "cbr_lightning/wikitext-103-tokenized"  # tokenized save path
    tokenizer_path = "./tokenizer.pkl"
    
    # Configuration
    sequence_length = 128
    stride = 128
    text_chunk_size = 1000  # Reduce if memory issues
    batch_size = 10000      # Reduce if memory issues
    
    # Estimate memory usage first
    estimate_memory_usage(data_path, sequence_length, stride)
    
    print("\n" + "="*50)
    print("🚀 Starting memory-efficient preprocessing...")
    print("="*50)
    
    # Check if tokenizer already exists
    tokenizer_path_obj = Path(tokenizer_path)
    if tokenizer_path_obj.exists():
        print(f"\n📚 Loading existing tokenizer from {tokenizer_path}...")
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"✅ Tokenizer loaded with {len(tokenizer.word2idx)} tokens")
    else:
        # Build new tokenizer IDENTICALLY to original script
        print("\n📚 Building new tokenizer (identical to original script)...")
        tokenizer = WordTokenizer(vocab_size=50000, min_freq=2)
        
        raw = datasets.load_from_disk(data_path)
        
        # USE IDENTICAL VOCAB BUILDING METHOD AS ORIGINAL SCRIPT
        build_vocab_identical(raw['train'], tokenizer)
        
        tokenizer.save(tokenizer_path)
        print(f"💾 Tokenizer saved to {tokenizer_path}")
        
        # Clear raw dataset from memory
        del raw
        gc.collect()
    
    # Preprocess & save efficiently
    print(f"\n🔄 Starting efficient preprocessing...")
    ds = preprocess_and_save_efficient(
        data_path=data_path,
        tokenizer=tokenizer,
        out_path=out_path,
        sequence_length=sequence_length,
        stride=stride,
        text_chunk_size=text_chunk_size,
        batch_size=batch_size
    )
    
    print(f"\n🎉 Processing complete!")
    print(f"📁 Output saved to: {out_path}")
    
    # Final memory cleanup
    del ds, tokenizer
    gc.collect()