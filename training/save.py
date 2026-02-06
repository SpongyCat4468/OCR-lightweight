import json
import os
from tqdm import tqdm
import dataset as dt

def save_dataset_alphabet(dataset, save_path="alphabet.json"):
    """
    Iterates through the entire dataset once to find every unique character
    and saves them to a fixed JSON file.
    """
    all_chars = set()
    print(f"Extracting alphabet from {len(dataset)} samples...")
    
    # Iterate through labels to find unique characters
    for i in tqdm(range(len(dataset))):
        # Assuming your dataset __getitem__ returns (image, label)
        _, label = dataset[i]
        all_chars.update(list(label))
    
    # Sort them to ensure index 1 is always the same character
    alphabet = sorted(list(all_chars))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(alphabet, f)
    
    print(f"\nSuccess! Alphabet saved to {save_path}")
    print(f"Total unique characters: {len(alphabet)}")
    print(f"Characters: {''.join(alphabet)}")

# Usage:
# synth_dataset = SynthTextDataset(...) 
# save_dataset_alphabet(synth_dataset)

dataset = dt.CroppedSynthTextDataset("./SynthText_Crops")
save_dataset_alphabet(dataset)