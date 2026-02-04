import os
import scipy.io
from PIL import Image
from tqdm import tqdm

def verify_dataset(root_dir, crops_dir):
    manifest_path = os.path.join(crops_dir, "labels.txt")
    
    if not os.path.exists(manifest_path):
        print("❌ Error: labels.txt not found!")
        return

    # 1. Load the manifest entries
    print("Reading manifest...")
    manifest_entries = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                manifest_entries[parts[0]] = parts[1]

    print(f"Total entries in manifest: {len(manifest_entries)}")

    # 2. Cross-reference with .mat file (Completeness check)
    print("Loading .mat file for total count verification...")
    mat_data = scipy.io.loadmat(os.path.join(root_dir, 'gt.mat'))
    texts = mat_data['txt'][0]
    
    total_expected_words = 0
    for img_text_array in texts:
        for line in img_text_array:
            total_expected_words += len(line.split())
    
    print(f"Total words expected from .mat: {total_expected_words}")
    print(f"Missing from manifest: {total_expected_words - len(manifest_entries)}")

    # 3. Physical File Integrity Check (The "Health Scan")
    print("\nStarting physical file scan (checking for corruption/missing files)...")
    missing_files = []
    corrupted_files = []
    
    # We check a sample of 10,000 or the whole thing? 
    # For a full check, we loop through all entries:
    for img_name in tqdm(manifest_entries.keys(), desc="Checking files"):
        img_path = os.path.join(crops_dir, img_name)
        
        if not os.path.exists(img_path):
            missing_files.append(img_name)
            continue
            
        # Optional: Deep integrity check (opens the file to check for corruption)
        # Warning: This makes the script slower. Remove the try/except if you only want to check existence.
        try:
            with Image.open(img_path) as img:
                img.verify() # Verify it's a valid image without loading whole pixels
        except Exception:
            corrupted_files.append(img_name)

    # 4. Final Report
    print("\n--- FINAL REPORT ---")
    if not missing_files and not corrupted_files:
        print("✅ SUCCESS: All manifest files exist and are healthy.")
    else:
        print(f"⚠️ Found {len(missing_files)} missing files.")
        print(f"⚠️ Found {len(corrupted_files)} corrupted files.")
        
        if missing_files:
            print(f"First 5 missing: {missing_files[:5]}")

if __name__ == "__main__":
    # Ensure these paths match your setup
    # verify_dataset('./SynthText', './SynthText_Crops')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, "training.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("hi\n")
        f.flush()
        os.fsync(f.fileno())