import os
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def crop_synthtext(root_dir, output_dir):
    mat_path = os.path.join(root_dir, 'gt.mat')
    mat_data = scipy.io.loadmat(mat_path)
    
    imnames = mat_data['imnames'][0]
    texts = mat_data['txt'][0]
    wordBB = mat_data['wordBB'][0] # The bounding boxes
    
    os.makedirs(output_dir, exist_ok=True)
    manifest = open(os.path.join(output_dir, "labels.txt"), "w", encoding="utf-8")
    
    count = 0
    for i in tqdm(range(len(imnames)), desc="Cropping"):
        img_path = os.path.join(root_dir, imnames[i][0])
        if not os.path.exists(img_path): continue
        
        # Load image once
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        
        # Get all words and their boxes in this image
        img_texts = []
        for line in texts[i]:
            img_texts.extend([t.strip() for t in line.split()])
            
        boxes = wordBB[i]
        # SynthText boxes are (2, 4, num_words)
        if len(boxes.shape) == 2: # Only one word
            boxes = boxes[:, :, np.newaxis]

        for j in range(min(len(img_texts), boxes.shape[2])):
            text = img_texts[j]
            if not text: continue
            
            # Get 4 corners of the box
            pts = boxes[:, :, j].T # (4, 2)
            
            # Get axis-aligned crop (min/max of corners)
            min_x = max(0, int(np.min(pts[:, 0])))
            min_y = max(0, int(np.min(pts[:, 1])))
            max_x = min(img_width, int(np.max(pts[:, 0])))
            max_y = min(img_height, int(np.max(pts[:, 1])))
            
            if max_x - min_x < 2 or max_y - min_y < 2: continue
            
            # Crop and save
            word_img = img.crop((min_x, min_y, max_x, max_y))
            save_name = f"word_{count}.jpg"
            word_img.save(os.path.join(output_dir, save_name))
            
            # Write label to manifest
            manifest.write(f"{save_name}\t{text}\n")
            count += 1
            
    manifest.close()
    print(f"Finished! Saved {count} crops to {output_dir}")

# Run it
crop_synthtext('./SynthText', './SynthText_Crops')