import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class Dataset(Dataset):
    """Dataset that loads all available SynthText data with minimal filtering"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        mat_path = os.path.join(root_dir, 'gt.mat')
        print(f"Loading annotations from {mat_path}...")
        
        mat_data = scipy.io.loadmat(mat_path)
        raw_image_names = mat_data['imnames'][0]
        raw_texts = mat_data['txt'][0]
        
        self.samples = []
        
        print("Processing all dataset samples...")
        for idx in tqdm(range(len(raw_image_names)), desc="Loading"):
            img_name = raw_image_names[idx][0]
            img_path = os.path.join(root_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            text_array = raw_texts[idx]
            for text_region in text_array:
                text_list = [t.strip() for t in text_region.splitlines()]
                
                for text in text_list:
                    text = ' '.join(text.split()) 
                    
                    if text: 
                        self.samples.append({
                            'image_path': img_path,
                            'text': text
                        })
        
        print(f"\nSuccessfully loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (128, 32), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['text']
    
class CroppedSynthTextDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        manifest_path = os.path.join(root_dir, "labels.txt")
        print(f"Loading manifest from {manifest_path}...")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.samples.append({
                        'img_name': parts[0],
                        'text': parts[1]
                    })
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['img_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If an image is missing, return a blank one to avoid crashing the batch
            image = Image.new('RGB', (128, 32), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, sample['text']