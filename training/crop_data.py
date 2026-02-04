import os
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_batch(batch_data):
    root_dir, output_dir, scene_data = batch_data
    results = []
    
    for img_name, img_texts, boxes, start_idx in scene_data:
        # Check if the FIRST word of this scene exists. 
        # If it does, we assume the whole scene was processed.
        if os.path.exists(os.path.join(output_dir, f"word_{start_idx}.jpg")):
            # Still need to build the manifest lines for the text file
            for j, text in enumerate(img_texts):
                if text.strip():
                    results.append(f"word_{start_idx + j}.jpg\t{text.strip()}")
            continue 

        # ... rest of your cropping logic ...

def clean_manifest(file_path):
    print("Cleaning duplicates...")
    lines_seen = set()
    outfile = open(file_path + ".tmp", "w", encoding="utf-8")
    for line in open(file_path, "r", encoding="utf-8"):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    os.replace(file_path + ".tmp", file_path)
    print("Done!")

def fast_crop_synthtext(root_dir, output_dir, num_workers=10):
    print("Loading .mat file (This takes a minute)...")
    mat_data = scipy.io.loadmat(os.path.join(root_dir, 'gt.mat'))
    
    imnames = mat_data['imnames'][0]
    texts = mat_data['txt'][0]
    wordBB = mat_data['wordBB'][0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group images into batches to feed workers efficiently
    print("Preparing batches...")
    all_scenes = []
    curr_idx = 0
    for i in range(len(imnames)):
        img_texts = []
        for line in texts[i]:
            img_texts.extend([t.strip() for t in line.split()])
        all_scenes.append((imnames[i][0], img_texts, wordBB[i], curr_idx))
        curr_idx += len(img_texts)

    # Chunking: Give each worker 100 images at a time to reduce communication overhead
    chunk_size = 100
    chunks = [(root_dir, output_dir, all_scenes[i:i + chunk_size]) 
              for i in range(0, len(all_scenes), chunk_size)]

    print(f"Starting cropping with {num_workers} workers...")
    manifest_path = os.path.join(output_dir, "labels.txt")
    
    # Use 'a' to append, but we might get duplicates if resuming. 
    # That's okay for now; we can clean the .txt later.
    with open(manifest_path, "a", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result_list in tqdm(executor.map(process_batch, chunks), total=len(chunks)):
                if result_list:
                    f.write("\n".join(result_list) + "\n")

if __name__ == "__main__":
    #fast_crop_synthtext('./SynthText', './SynthText_Crops', num_workers=10)
    clean_manifest('./SynthText_Crops/labels.txt')



