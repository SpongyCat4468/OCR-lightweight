import json

class CharsetMapper:
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = ''  # CTC blank
        
        self.num_classes = len(self.chars) + 1
        print(f"Locked Charset size: {self.num_classes}")

    @classmethod
    def from_dataset(cls, dataset):
        all_chars = set()
        for sample in dataset.samples:
            all_chars.update(sample['text'])
        return cls(list(all_chars))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            chars = json.load(f)
        return cls(chars)

    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices if idx != 0])