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
from PIL import Image
from torch.utils.data import ConcatDataset

DATASET_PATH = "./IIIT5K"
REAL_DATA_PATH = "./IIIT5K"
SYNTH_DATA_PATH = "SynthText_Crops"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
EPOCHS = 300
MODEL_DIR = './model'
LEARNING_RATE = 0.000005 
USE_SCHEDULER = True
IMG_HEIGHT = 32
IMG_WIDTH = 128

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
    aligner = AlignCollate(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    train_transform = transforms.Compose([
        aligner,
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    real_dataset = dt.IIIT5KDataset(REAL_DATA_PATH)
    synth_dataset = dt.CroppedSynthTextDataset(SYNTH_DATA_PATH) 
    
    real_dataset.transform = train_transform
    synth_dataset.transform = train_transform

    combined_dataset = ConcatDataset([real_dataset, synth_dataset])
    print(f"Total Mixed Images: {len(combined_dataset)} ({len(real_dataset)} Real, {len(synth_dataset)} Synth)")

    charset = CharsetMapper.load("alphabet.json")

    train_size = int(0.95 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    collate_fn = CollateFn(charset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    ) 

    model = CRNN(3, IMG_HEIGHT, IMG_WIDTH, charset.num_classes).to(DEVICE)
    
    checkpoint_path, start_epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Resuming mixed training from {checkpoint_path}")

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, 
                                epoch, EPOCHS, len(train_loader))
        
        if scheduler:
            scheduler.step(train_loss)
            
        model_path = os.path.join(MODEL_DIR, f'crnn_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)

        with open(file="./training_finetune.txt", mode='a') as f:
            f.write(f"Epoch {epoch} | Loss: {train_loss}\n")