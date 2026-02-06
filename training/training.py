import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from charset import CharsetMapper
import dataset as dt
from model import CRNN
import os
import glob
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image

# --- UPDATED CONSTANTS ---
DATASET_PATH = "./IIIT5K"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128 
EPOCHS = 100      # Increased for full fine-tuning
MODEL_DIR = './model'
LEARNING_RATE = 0.000005 # Lowered for full model training
USE_SCHEDULER = True
IMG_HEIGHT = 32
IMG_WIDTH = 128

# ... [CollateFn and AlignCollate classes remain the same] ...

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

class AlignCollate:
    def __init__(self, img_height=32, img_width=128):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, image):
        w, h = image.size
        aspect_ratio = w / h
        new_w = int(self.img_height * aspect_ratio)
        new_w = min(new_w, self.img_width)
        img = image.resize((new_w, self.img_height), Image.BILINEAR)
        final_img = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        left_pad = (self.img_width - new_w) // 2
        final_img.paste(img, (left_pad, 0))
        return final_img

# ... [train_epoch and find_latest_checkpoint remain the same] ...

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs, batches_per_epoch):
    model.train()
    total_loss = 0
    batch_count = 0
    dataloader_iter = iter(dataloader)
    pbar = tqdm(range(batches_per_epoch), desc=f'Epoch {epoch}/{total_epochs}')
    for _ in pbar:
        try:
            images, targets, target_lengths = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            images, targets, target_lengths = next(dataloader_iter)
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        batch_count += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/batch_count:.4f}'})
    return total_loss / batch_count

def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, 'crnn_epoch_*.pt'))
    if not checkpoints: return None, 0
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    return os.path.join(model_dir, f'crnn_epoch_{latest_epoch}.pt'), latest_epoch

if __name__ == "__main__":
    dataset = dt.IIIT5KDataset(DATASET_PATH)
    aligner = AlignCollate(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    transform = transforms.Compose([
        aligner,
        # Add these three for "Robustness"
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handles lighting
        transforms.RandomGrayscale(p=0.1),                   # Handles color fading
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)), # Handles tilts
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset.transform = transform
    charset = CharsetMapper.load("alphabet.json")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collate_fn = CollateFn(charset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True) 
    
    BATCHES_PER_EPOCH = len(train_loader)

    model = CRNN(
        img_channel=3,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=charset.num_classes
    ).to(DEVICE)
    
    # --- STEP 1 & 2: LOAD CHECKPOINT AND ENSURE UNFROZEN ---
    checkpoint_path, start_epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded {checkpoint_path}. Starting unfreezed fine-tuning.")
    
    # Ensure all parameters are set to train (requires_grad = True)
    for param in model.parameters():
        param.requires_grad = True
    print("Full model unfrozen. CNN is now training.")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # --- STEP 3: OPTIMIZER WITH ALL PARAMETERS ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) if USE_SCHEDULER else None

    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, 
                                epoch, EPOCHS, BATCHES_PER_EPOCH)
        
        if scheduler:
            scheduler.step(train_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f} - LR: {current_lr:.6f}"
        print(log_msg)
        
        with open("training_finetune.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        model_path = os.path.join(MODEL_DIR, f'crnn_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)

    print("Fine-tuning complete!")