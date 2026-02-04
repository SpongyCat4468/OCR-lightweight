import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset as dt
from charset import CharsetMapper
from model import CRNN
from torchvision import transforms

# Use the same constants as your training script
IMG_HEIGHT = 32
IMG_WIDTH = 128
DATASET_PATH = "./SynthText"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import your CollateFn from your training file or paste it here
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

def find_max_batch_size(dataset, charset, device, start_batch=32, max_batch=1024):
    batch_size = start_batch
    best_batch = start_batch
    
    # 1. Initialize your specific CollateFn
    collate_fn = CollateFn(charset)
    
    model = CRNN(img_channel=3, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, 
                 num_class=charset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    print(f"Searching for optimal batch size on {device}...")

    while batch_size <= max_batch:
        try:
            # 2. ADDED collate_fn=collate_fn HERE
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=2, 
                collate_fn=collate_fn  # This fix ensures we get 3 values back
            )
            
            # Now this will correctly unpack into 3 variables
            images, targets, target_lengths = next(iter(loader))
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Batch size {batch_size} works.")
            best_batch = batch_size
            batch_size *= 2 
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch size {batch_size} is too large.")
                torch.cuda.empty_cache()
                break
            else:
                raise e
                
    return best_batch

if __name__ == "__main__":
    # Setup Dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_dataset = dt.Dataset(DATASET_PATH)
    raw_dataset.transform = transform
    charset = CharsetMapper(raw_dataset)
    
    optimal_batch = find_max_batch_size(raw_dataset, charset, DEVICE)
    print(f"\nSuggested BATCH_SIZE for your training script: {optimal_batch}")