import json 

class WordTokenizer:
    """Simple word-level tokenizer for WikiText"""
    def __init__(self, list_of_texts, vocab_size=50000):
        tokens = []
        for text in list_of_texts:
            tokens.extend(text.split())
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)
        self.itos = ["<unk>"] + [tok for tok, _ in most_common]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(tok, 0) for tok in text.split()]

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids])

    def save(self, path):
        """Save tokenizer to file"""
        with open(path, 'w') as f:
            json.dump({
                'itos': self.itos,
                'stoi': self.stoi,
                'vocab_size': self.vocab_size
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
        return tokenizer