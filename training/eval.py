'''import torch
from tqdm import tqdm
from model import CRNN
from charset import CharsetMapper
from training import AlignCollate
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import os

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/crnn_epoch_100.pt'
TEST_DATA_PATH = './IIIT5K/test' # Adjust path to your test folder
IMG_HEIGHT, IMG_WIDTH = 32, 128

def evaluate():
    charset = CharsetMapper.load("alphabet.json")
    model = CRNN(3, IMG_HEIGHT, IMG_WIDTH, charset.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    aligner = AlignCollate(IMG_HEIGHT, IMG_WIDTH)
    transform = transforms.Compose([
        aligner,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    mat_path = os.path.join('./IIIT5K', 'testdata.mat')
    test_data = sio.loadmat(mat_path)['testdata'][0]
    test_samples = [] 
    for sample in test_data:
        img_name = sample[0][0]
        label = sample[1][0]
        full_img_path = os.path.join('./IIIT5K', img_name)
        test_samples.append((full_img_path, label))
    total_words = len(test_samples)
    correct_words = 0

    print("Evaluating Test Set...")
    '''
'''for img_path, gt_text in tqdm(test_samples):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        # Greedy Decode
        out_indices = outputs.argmax(2).squeeze(1).tolist()
        
        pred_text = ""
        prev_idx = -1
        for idx in out_indices:
            if idx != 0 and idx != prev_idx:
                pred_text += charset.chars[idx - 1]
            prev_idx = idx

    if pred_text.upper() == gt_text.upper():
        correct_words += 1
    total_words += 1'''

'''
print("Evaluating Test Set...")
for i, (img_path, gt_text) in enumerate(tqdm(test_samples)):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        out_indices = outputs.argmax(2).squeeze(1).tolist()
        
        # THE CRITICAL DECODE CHECK
        pred_text = ""
        prev_idx = -1
        for idx in out_indices:
            if idx != 0 and idx != prev_idx:
                # CHECK THIS: If your charset.chars starts at 'A', 
                # and idx 1 is 'A', then idx-1 is correct.
                # If idx 1 is 'B', you have a shift.
                if (idx - 1) < len(charset.chars):
                    pred_text += charset.chars[idx - 1]
            prev_idx = idx

    # DEBUG: Print the first 5 to see the shift
    if i < 5:
        print(f"\nTarget: {gt_text} | Pred: {pred_text} | Indices: {out_indices[:10]}")

    if pred_text.upper() == gt_text.upper():
        correct_words += 1
accuracy = (correct_words / total_words) * 100
print(f"\n--- Evaluation Results ---")
print(f"Total Images: {total_words}")
print(f"Word Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
evaluate()'''

import torch
from charset import CharsetMapper
from model import CRNN
from PIL import Image
import torchvision.transforms as transforms

# Load the WORKING checkpoint (before training)
checkpoint = './model/crnn_epoch_1.pt'  # Use the epoch that worked
charset = CharsetMapper.load("alphabet.json")

model = CRNN(3, 32, 128, charset.num_classes)
model.load_state_dict(torch.load(checkpoint))
model.eval()

# Test on known working images
test_cases = [
    ("./input/text.for.jpg", "FOR"),
    ("./input/text.smile.png", "SMILE")
]

def original_decode(outputs, charset):
    """Put your ORIGINAL working decoder here"""
    # Copy from your backup or git history
    pass

for img_path, expected in test_cases:
    # Load and preprocess
    img = Image.open(img_path)
    # ... transform ...
    
    outputs = model(img)
    pred = original_decode(outputs, charset)
    
    print(f"Expected: '{expected}' | Got: '{pred}' | {'✅' if pred==expected else '❌'}")