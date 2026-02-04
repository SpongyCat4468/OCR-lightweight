class CharsetMapper:
    """Maps characters to indices for the model"""
    
    def __init__(self, dataset):
        # Extract all unique characters
        all_chars = set()
        for sample in dataset.samples:
            all_chars.update(sample['text'])
        
        chars = sorted(list(all_chars))
        
        # Create mappings (0 is reserved for CTC blank)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
        self.idx_to_char[0] = ''  # CTC blank
        
        self.num_classes = len(chars) + 1  # +1 for blank
        
        print(f"Charset size: {self.num_classes} (including blank)")
        print(f"Characters: {''.join(chars[:50])}..." if len(chars) > 50 else f"Characters: {''.join(chars)}")
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
    
    def decode(self, indices):
        """Convert indices to text"""
        chars = []
        for idx in indices:
            if idx == 0:  # CTC blank
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)