import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from charset import CharsetMapper
import dataset as dt
from model import CRNN
import os
import glob
from tqdm import tqdm
import random
from PIL import Image

DATASET_PATH = "./IIIT5K"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
MODEL_DIR = './model'
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

def decode_predictions(outputs, charset):
    """Decode CTC outputs to text. outputs: (seq_len, batch, num_class)"""
    predictions = []
    outputs = outputs.permute(1, 0, 2)  # (batch, seq_len, num_class)
    
    for output in outputs:
        pred_indices = output.argmax(dim=1)
        
        decoded = []
        prev_idx = None
        for idx in pred_indices:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                decoded.append(idx)
            prev_idx = idx
        
        text = charset.decode(decoded)
        predictions.append(text)
    
    return predictions

def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, 'crnn_epoch_*.pt'))
    if not checkpoints:
        return None, 0
    
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(model_dir, f'crnn_epoch_{latest_epoch}.pt')
    
    return latest_checkpoint, latest_epoch

def test_model(model, dataloader, charset, device, num_samples=10):
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"Sample Predictions (showing {num_samples} random samples)")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)  # (seq_len, batch, num_class)
            
            predictions = decode_predictions(outputs, charset)
            
            ground_truths = []
            for i in range(len(targets)):
                target = targets[i][:target_lengths[i]].tolist()
                ground_truths.append(charset.decode(target))
            
            indices = random.sample(range(len(predictions)), min(num_samples, len(predictions)))
            
            for idx in indices:
                print(f"Ground Truth: '{ground_truths[idx]}'")
                print(f"Prediction:   '{predictions[idx]}'")
                print(f"Match: {ground_truths[idx] == predictions[idx]}")
                print("-" * 80)
            
            break

def calculate_accuracy(model, dataloader, charset, device):
    model.eval()
    correct = 0
    total = 0
    
    print("\nCalculating accuracy on validation set...")
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)  # (seq_len, batch, num_class)
            
            predictions = decode_predictions(outputs, charset)
            
            for i in range(len(targets)):
                target = targets[i][:target_lengths[i]].tolist()
                ground_truth = charset.decode(target)
                
                if predictions[i] == ground_truth:
                    correct += 1
                total += 1
    
    accuracy = (correct / total) * 100
    return accuracy, correct, total

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = dt.IIIT5KDataset(DATASET_PATH)

    # Create charset FIRST, before applying any transforms
    # CharsetMapper needs to iterate through raw text labels
    print("Building character set...")
    charset = CharsetMapper(dataset)
    print(f"Charset size: {charset.num_classes}")
    
    # Now apply transforms to the dataset
    aligner = AlignCollate(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    transform = transforms.Compose([
        aligner,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])  
    dataset.transform = transform

    # Split dataset
    from torch.utils.data import random_split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Validation size: {len(val_dataset)}")

    # Create dataloader
    collate_fn = CollateFn(charset)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=collate_fn, num_workers=4)

    # Initialize model
    model = CRNN(
        img_channel=3,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=charset.num_classes
    ).to(DEVICE)
    
    # Load checkpoint
    checkpoint_path, epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        print(f"\nLoading model from epoch {epoch}: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Model loaded successfully!")
    else:
        print("\nNo checkpoint found! Please train the model first.")
        exit()

    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test model
    test_model(model, val_loader, charset, DEVICE, num_samples=10)
    
    # Calculate accuracy
    accuracy, correct, total = calculate_accuracy(model, val_loader, charset, DEVICE)
    
    print(f"\n{'='*80}")
    print(f"Validation Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*80}\n")