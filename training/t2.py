import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_crops(crops_dir, num_samples=20):
    manifest_path = os.path.join(crops_dir, "labels.txt")
    
    if not os.path.exists(manifest_path):
        print("❌ Error: labels.txt not found!")
        return

    # Load all entries
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                samples.append(parts)

    if not samples:
        print("❌ Manifest is empty!")
        return

    # Pick random samples
    random_samples = random.sample(samples, min(num_samples, len(samples)))

    # Set up the plot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))

    for i, (img_name, text) in enumerate(random_samples):
        img_path = os.path.join(crops_dir, img_name)
        try:
            img = Image.open(img_path)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(f"Label: {text}\nFile: {img_name}", fontsize=10)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_crops('./SynthText_Crops')