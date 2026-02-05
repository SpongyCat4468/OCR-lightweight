from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import torch
import glob
from training.model import CRNN
import training.dataset as dt
from training.charset import CharsetMapper
from torchvision import transforms

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r'C:\Users\User\Desktop\Coding\Python\PyTorch\OCR lightweight\uploads'
DATASET_PATH = r'C:\Users\User\Desktop\Coding\Python\PyTorch\OCR lightweight\SynthText_Crops'
MODEL_DIR = r'C:\Users\User\Desktop\Coding\Python\PyTorch\OCR lightweight\model'

IMG_HEIGHT = 32
IMG_WIDTH = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ========== LOAD MODEL AND CHARSET ONCE AT STARTUP ==========
print("Loading charset...")
temp_dataset = dt.CroppedSynthTextDataset(DATASET_PATH)
charset = CharsetMapper(temp_dataset)

print("Initializing model...")
model = CRNN(
    img_channel=3,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    num_class=charset.num_classes
).to(DEVICE)

# Find and load the latest checkpoint
def find_latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, 'crnn_epoch_*.pt'))
    if not checkpoints:
        return None, 0
    
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(model_dir, f'crnn_epoch_{latest_epoch}.pt')
    
    return latest_checkpoint, latest_epoch

checkpoint_path, epoch = find_latest_checkpoint(MODEL_DIR)
if checkpoint_path:
    print(f"Loading model from epoch {epoch}: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print("Model loaded successfully!")
else:
    print("WARNING: No checkpoint found! Model will use random weights.")

model.eval()

# Define transform once
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])
# ============================================================

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
            if idx != 0 and idx != prev_idx:  # 0 is blank token in CTC
                decoded.append(idx)
            prev_idx = idx
        
        text = charset.decode(decoded)
        predictions.append(text)
    
    return predictions

def image_to_text(image):
    """Process a single image and return predicted text"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)  # (seq_len, 1, num_class)
        
        # Decode prediction
        prediction = decode_predictions(outputs, charset)
        return prediction[0]
    
    except Exception as e:
        print(f"Error in image_to_text: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recognize', methods=['POST'])
def recognize_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = None
    try:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Open and process image
        image = Image.open(filepath)
        text = image_to_text(image)
        
        return jsonify({'text': text})
    
    except Exception as e:
        print(f"Error in recognize_text: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up: delete the file after processing
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == "__main__":
    print(f"\nFlask app starting on {DEVICE}")
    print(f"Charset size: {charset.num_classes}")
    print("="*80)
    app.run(debug=True, use_reloader=False)