"""
Modern Windows Desktop App for Text Recognition
Uses CustomTkinter for a sleek, modern interface

Installation:
    pip install customtkinter pillow torch torchvision

Usage:
    python windows_app.py
"""

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import threading
import os
import torch
from torchvision import transforms
import glob

print("Loading model...")

# Import your modules
import sys
sys.path.append('./training')  # Adjust path if needed
from training.charset import CharsetMapper
import training.dataset as dt
from training.model import CRNN

# Setup
DATASET_PATH = r"C:\Users\User\.cache\doctr\datasets\SynthText\SynthText"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './model'
IMG_HEIGHT = 32
IMG_WIDTH = 128

# Load charset
temp_dataset = dt.Dataset(DATASET_PATH)
charset = CharsetMapper(temp_dataset)

# Load model
model = CRNN(
    img_channel=3,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    num_class=charset.num_classes
).to(DEVICE)

# Find and load latest checkpoint
checkpoints = glob.glob(os.path.join(MODEL_DIR, 'crnn_epoch_*.pth'))
if checkpoints:
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    checkpoint_path = os.path.join(MODEL_DIR, f'crnn_epoch_{latest_epoch}.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"Loaded model from epoch {latest_epoch}")
else:
    print("WARNING: No checkpoint found! Using untrained model.")

model.eval()

# Setup transform
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])

print("Model loaded successfully!")


def decode_prediction(output, charset):
    """Decode CTC output to text. output: (seq_len, 1, num_class)"""
    output = output.permute(1, 0, 2)[0]  # (seq_len, num_class)
    pred_indices = output.argmax(dim=1)  # (seq_len,)
    
    # Remove consecutive duplicates and blanks
    decoded = []
    prev_idx = None
    for idx in pred_indices:
        idx = idx.item()
        if idx != 0 and idx != prev_idx:  # 0 is blank
            decoded.append(idx)
        prev_idx = idx
    
    return charset.decode(decoded)


def recognize_text_from_image(image):
    """
    Process image and return recognized text
    
    Args:
        image: PIL Image object
    
    Returns:
        str: Recognized text
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)  # (seq_len, 1, num_class)
        
        # Decode
        recognized_text = decode_prediction(outputs, charset)
        
        result = f"""✓ Image processed successfully!

Image Size: {image.size}
Image Mode: {image.mode}
Device: {DEVICE}

Recognized Text:
{recognized_text}"""
        
        return result
        
    except Exception as e:
        return f"Error during recognition: {str(e)}"


class TextRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Text Recognition")
        self.geometry("1000x700")
        
        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue") 
        
        self.current_image = None
        self.current_image_path = None
        self.photo_image = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all UI components"""
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Text Recognition",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Select Image",
            command=self.select_image,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.upload_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.recognize_btn = ctk.CTkButton(
            self.sidebar,
            text="Recognize Text",
            command=self.process_image,
            height=40,
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.recognize_btn.grid(row=2, column=0, padx=20, pady=10)
        
        self.clear_btn = ctk.CTkButton(
            self.sidebar,
            text="Clear",
            command=self.clear_all,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            border_width=2
        )
        self.clear_btn.grid(row=3, column=0, padx=20, pady=10)
        
        self.theme_label = ctk.CTkLabel(
            self.sidebar,
            text="Appearance:",
            font=ctk.CTkFont(size=12)
        )
        self.theme_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        
        self.theme_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Dark Mode",
            command=self.toggle_theme,
            onvalue="dark",
            offvalue="light"
        )
        self.theme_switch.grid(row=6, column=0, padx=20, pady=(0, 20))
        self.theme_switch.select()
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.preview_frame = ctk.CTkFrame(self.main_frame)
        self.preview_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        self.preview_label_text = ctk.CTkLabel(
            self.preview_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.preview_label_text.pack(pady=(10, 5))
        
        self.image_label = ctk.CTkLabel(
            self.preview_frame,
            text="No image selected\n\nClick 'Select Image' to get started",
            width=700,
            height=300,
            fg_color=("gray75", "gray25")
        )
        self.image_label.pack(padx=20, pady=(0, 20))
        
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        self.results_header = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.results_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.results_header.grid_columnconfigure(0, weight=1)
        
        self.results_label = ctk.CTkLabel(
            self.results_header,
            text="Recognized Text",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.results_label.grid(row=0, column=0, sticky="w")
        
        self.copy_btn = ctk.CTkButton(
            self.results_header,
            text="Copy",
            command=self.copy_text,
            width=100,
            height=30
        )
        self.copy_btn.grid(row=0, column=1, padx=10)
        
        self.save_btn = ctk.CTkButton(
            self.results_header,
            text="Save",
            command=self.save_text,
            width=100,
            height=30
        )
        self.save_btn.grid(row=0, column=2)
        
        self.output_text = ctk.CTkTextbox(
            self.results_frame,
            font=ctk.CTkFont(size=13),
            wrap="word"
        )
        self.output_text.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.status_label.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="ew")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display the selected image"""
        try:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path)
            
            display_image = self.current_image.copy()
            display_image.thumbnail((700, 300), Image.Resampling.LANCZOS)
            
            self.photo_image = ctk.CTkImage(
                light_image=display_image,
                dark_image=display_image,
                size=display_image.size
            )
            
            self.image_label.configure(image=self.photo_image, text="")
            
            self.recognize_btn.configure(state="normal")
            
            filename = os.path.basename(file_path)
            self.status_label.configure(text=f"Loaded: {filename}")
            
        except Exception as e:
            self.show_error("Error Loading Image", str(e))
    
    def process_image(self):
        """Process the image with the model"""
        if self.current_image is None:
            self.show_error("No Image", "Please select an image first")
            return
        
        self.recognize_btn.configure(state="disabled")
        self.status_label.configure(text="Processing...")
        self.output_text.delete("0.0", "end")
        self.output_text.insert("0.0", "Processing image, please wait...\n")
        
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True
        thread.start()
    
    def _process_thread(self):
        """Background thread for processing"""
        try:
            recognized_text = recognize_text_from_image(self.current_image)
            
            self.after(0, self._update_results, recognized_text, None)
            
        except Exception as e:
            self.after(0, self._update_results, None, str(e))
    
    def _update_results(self, text, error):
        """Update the results in the UI"""
        self.output_text.delete("0.0", "end")
        
        if error:
            self.output_text.insert("0.0", f"Error: {error}")
            self.status_label.configure(text="Error occurred")
            self.show_error("Processing Error", error)
        else:
            self.output_text.insert("0.0", text)
            self.status_label.configure(text="Processing complete ✓")
        
        self.recognize_btn.configure(state="normal")
    
    def copy_text(self):
        """Copy text to clipboard"""
        try:
            text = self.output_text.get("0.0", "end-1c")
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_label.configure(text="Copied to clipboard ✓")
        except Exception as e:
            self.show_error("Copy Error", str(e))
    
    def save_text(self):
        """Save text to file"""
        try:
            text = self.output_text.get("0.0", "end-1c")
            if not text.strip():
                self.show_error("Nothing to Save", "No text to save")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_label.configure(text=f"Saved to {os.path.basename(file_path)} ✓")
        except Exception as e:
            self.show_error("Save Error", str(e))
    
    def clear_all(self):
        """Clear all data"""
        self.current_image = None
        self.current_image_path = None
        self.photo_image = None
        self.image_label.configure(
            image=None,
            text="No image selected\n\nClick 'Select Image' to get started"
        )
        self.output_text.delete("0.0", "end")
        self.recognize_btn.configure(state="disabled")
        self.status_label.configure(text="Cleared")
    
    def toggle_theme(self):
        """Toggle between light and dark mode"""
        current_mode = ctk.get_appearance_mode()
        new_mode = "light" if current_mode == "Dark" else "dark"
        ctk.set_appearance_mode(new_mode)
    
    def show_error(self, title, message):
        """Show error dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        
        label = ctk.CTkLabel(dialog, text=message, wraplength=350)
        label.pack(padx=20, pady=20)
        
        button = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        button.pack(pady=10)
        
        dialog.transient(self)
        dialog.grab_set()


def main():
    """Run the application"""
    print("=" * 60)
    print("Starting Text Recognition App...")
    print("=" * 60)
    
    app = TextRecognitionApp()
    app.mainloop()


if __name__ == '__main__':
    main()