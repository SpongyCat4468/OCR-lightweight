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

DATASET_PATH = r"C:\Users\User\.cache\doctr\datasets\SynthText\SynthText"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128  # Increased from 64 for better gradient estimates
BATCHES_PER_EPOCH = 2500
EPOCHS = 100
MODEL_DIR = '../OCR lightweight/model'
LEARNING_RATE = 0.0005  # Reduced from 0.001
USE_SCHEDULER = True  # Add learning rate scheduling
IMG_HEIGHT = 32
IMG_WIDTH = 128

class CollateFn:
    def __init__(self, charset):
        self.charset = charset
    
    def __call__(self, batch):
        images, texts = zip(*batch)
        images = torch.stack(images, 0)
        
        text_encoded = [self.charset.encode(text) for text in texts]
        text_lengths = torch.LongTensor([len(t) for t in text_encoded])
        
        max_len = max(text_lengths)
        text_padded = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, encoded in enumerate(text_encoded):
            text_padded[i, :len(encoded)] = torch.LongTensor(encoded)
        
        return images, text_padded, text_lengths

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, batches_per_epoch):
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
        outputs = model(images)
        
        input_lengths = torch.full(
            size=(outputs.size(1),), 
            fill_value=outputs.size(0), 
            dtype=torch.long
        )
        
        loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
        
        # Skip batch if loss is nan or inf
        if torch.isnan(loss) or torch.isinf(loss):
            continue
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
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

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Batches per epoch: {BATCHES_PER_EPOCH}")
    print(f"Samples per epoch: {BATCHES_PER_EPOCH * BATCH_SIZE:,}")

    collate_fn = CollateFn(charset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=collate_fn, num_workers=4)

    model = CRNN(
        img_channel=3,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=charset.num_classes
    ).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    ) if USE_SCHEDULER else None

    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Learning rate: {LEARNING_RATE}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    checkpoint_path, start_epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        print(f"\nLoading checkpoint from epoch {start_epoch}: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        print("WARNING: Starting with lower learning rate. Consider restarting from scratch.")
    else:
        start_epoch = 0
        print("\nNo checkpoint found, starting from scratch")

    print(f"\nStarting training from epoch {start_epoch + 1}\n")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, 
                                epoch, EPOCHS, BATCHES_PER_EPOCH)
        
        if scheduler:
            scheduler.step(train_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f} - LR: {current_lr:.6f}"
        print(log_msg)

        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.txt')
        with open(log_path, 'a') as f:
            f.write(log_msg + '\n')
        
        model_path = os.path.join(MODEL_DIR, f'crnn_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}\n")

    print("Training complete!")