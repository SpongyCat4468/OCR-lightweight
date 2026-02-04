import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

# Import your existing classes
# Assuming these are in your local files:
from model import CRNN 
from charset import CharsetMapper

# --- CONFIGURATION ---
DATASET_PATH = "./SynthText_Crops"
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CroppedSynthTextDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        manifest_path = os.path.join(root_dir, "labels.txt")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.samples.append({'img_name': parts[0], 'text': parts[1]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['img_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='white')
        if self.transform:
            image = self.transform(image)
        return image, sample['text']

class CollateFn:
    def __init__(self, charset):
        self.charset = charset
    
    def __call__(self, batch):
        images, texts = zip(*batch)
        images = torch.stack(images, 0)
        text_encoded = [self.charset.encode(text) for text in texts]
        text_lengths = torch.IntTensor([len(t) for t in text_encoded])
        max_len = max(text_lengths)
        text_padded = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, encoded in enumerate(text_encoded):
            text_padded[i, :len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        return images, text_padded, text_lengths

def find_max_batch_size(dataset, charset, device):
    # Try these sizes in order
    test_sizes = [64, 128, 256, 512, 1024, 2048]
    best_batch = 0
    
    # Initialize real model components to fill VRAM
    model = CRNN(img_channel=3, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, 
                 num_class=charset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    collate_fn = CollateFn(charset)

    print(f"🚀 Testing GPU limits on {torch.cuda.get_device_name(0)}...")

    for batch_size in test_sizes:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=2, collate_fn=collate_fn)
            
            # Grab one batch
            images, targets, target_lengths = next(iter(loader))
            images, targets = images.to(device), targets.to(device)

            # --- DUMMY TRAINING STEP ---
            # Forward pass
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
            # Backward pass (This is where most OOMs happen!)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f" ✅ Batch Size {batch_size:4}: SUCCESS")
            best_batch = batch_size
            
            # Clean up memory for next test
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" ❌ Batch Size {batch_size:4}: OUT OF MEMORY")
                torch.cuda.empty_cache()
                break
            else:
                raise e
                
    return best_batch

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CroppedSynthTextDataset(DATASET_PATH, transform=transform)
    charset = CharsetMapper(dataset)
    
    max_bs = find_max_batch_size(dataset, charset, DEVICE)
    
    print("-" * 30)
    print(f"FINAL RESULT: Use BATCH_SIZE = {max_bs}")
    print("If you want safety, use 80% of this value.")
    print("-" * 30)