import torch
from torchvision import transforms
from charset import CharsetMapper
import dataset as dt
from model import CRNN
import os
import glob
from PIL import Image

DATASET_PATH = r"C:\Users\User\.cache\doctr\datasets\SynthText\SynthText"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './model'
INPUT_DIR = './input'
IMG_HEIGHT = 32
IMG_WIDTH = 128

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

def process_images(model, image_paths, transform, charset, device):
    model.eval()
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                outputs = model(image_tensor)  # (seq_len, 1, num_class)
                predictions = decode_predictions(outputs, charset)
                
                filename = os.path.basename(img_path)
                print(f"{filename}: {predictions[0]}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, f'*.{ext}')))
    
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"\nNo images found in {INPUT_DIR}")
        print("Please add images to the 'input' folder and run again.")
        exit()
    
    print(f"Found {len(image_paths)} images in {INPUT_DIR}\n")
    
    print("Loading charset...")
    temp_dataset = dt.Dataset(DATASET_PATH)
    charset = CharsetMapper(temp_dataset)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    
    model = CRNN(
        img_channel=3,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=charset.num_classes
    ).to(DEVICE)
    
    checkpoint_path, epoch = find_latest_checkpoint(MODEL_DIR)
    if checkpoint_path:
        print(f"Loading model from epoch {epoch}: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Model loaded successfully!\n")
    else:
        print("\nNo checkpoint found! Please train the model first.")
        exit()
    
    print(f"Device: {DEVICE}")
    print(f"Processing {len(image_paths)} images...\n")
    print("="*80)
    
    process_images(model, image_paths, transform, charset, DEVICE)
    
    print("="*80)