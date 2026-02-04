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

DATASET_PATH = "./SynthText"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
BATCHES_PER_EPOCH = 2500
EPOCHS = 100
MODEL_DIR = './model'
LEARNING_RATE = 0.0005
USE_SCHEDULER = True
IMG_HEIGHT = 32
IMG_WIDTH = 128

class CollateFn:
    def __init__(self, charset):
        self.charset = charset
    
    def __call__(self, batch):
        images, texts = zip(*batch)
        images = torch.stack(images, 0)
        
        # Streamlined: Pre-calculate lengths and prepare padded tensor immediately
        text_encoded = [self.charset.encode(text) for text in texts]
        text_lengths = torch.IntTensor([len(t) for t in text_encoded])
        max_len = max(text_lengths)
        
        text_padded = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, encoded in enumerate(text_encoded):
            text_padded[i, :len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        
        return images, text_padded, text_lengths

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
        
        # AMP: Use autocast for forward pass
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            
            input_lengths = torch.full(
                size=(outputs.size(1),), 
                fill_value=outputs.size(0), 
                dtype=torch.long
            )
            
            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
            
        # AMP: Scale loss and call backward
        scaler.scale(loss).backward()
        
        # AMP: Unscale before clipping gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # AMP: Step and Update
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batch_count += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/batch_count:.4f}'})
    
    return total_loss / batch_count

def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, 'crnn_epoch_*.pt'))
    if not checkpoints:
        return None, 0
    
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(model_dir, f'crnn_epoch_{latest_epoch}.pt')
    
    return latest_checkpoint, latest_epoch

if __name__ == "__main__":
    dataset = dt.Dataset(DATASET_PATH)

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    dataset.transform = transform
    charset = CharsetMapper(dataset)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collate_fn = CollateFn(charset)
    
    # Optimizing DataLoader for speed
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=collate_fn, num_workers=4, pin_memory=True,
                        persistent_workers=True)

    model = CRNN(
        img_channel=3,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=charset.num_classes
    ).to(DEVICE)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # AMP: Initialize GradScaler
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    ) if USE_SCHEDULER else None

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    checkpoint_path, start_epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        start_epoch = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        # Pass scaler to train_epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, 
                                epoch, EPOCHS, BATCHES_PER_EPOCH)
        
        if scheduler:
            scheduler.step(train_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f} - LR: {current_lr:.6f}"
        print(log_msg)
        with open("training.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        model_path = os.path.join(MODEL_DIR, f'crnn_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)

    print("Training complete!")