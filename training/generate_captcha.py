from captcha.image import ImageCaptcha
import random
import string
import os

# Configuration
OUTPUT_FOLDER = './captcha'
NUM_CAPTCHAS = 100  # Number of captcha images to generate
MIN_LENGTH = 4      # Minimum captcha text length
MAX_LENGTH = 8      # Maximum captcha text length
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60

# Character set for captcha (you can customize this)
CHARSET = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
# CHARSET = string.ascii_uppercase + string.digits  # Only uppercase and digits
# CHARSET = string.digits  # Only numbers

def generate_random_text(min_len=MIN_LENGTH, max_len=MAX_LENGTH):
    """Generate random text for captcha"""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(CHARSET, k=length))

def generate_captchas(num_images=NUM_CAPTCHAS, output_dir=OUTPUT_FOLDER):
    """Generate multiple captcha images and save them"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ImageCaptcha
    image_captcha = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    
    print(f"Generating {num_images} captcha images...")
    print(f"Output directory: {output_dir}")
    print(f"Character set: {CHARSET}")
    print(f"Text length: {MIN_LENGTH}-{MAX_LENGTH} characters")
    print("=" * 80)
    
    for i in range(num_images):
        # Generate random text
        captcha_text = generate_random_text()
        
        # Generate image
        image = image_captcha.generate_image(captcha_text)
        
        # Save image with text as filename
        filename = f"{captcha_text}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} captchas...")
    
    print("=" * 80)
    print(f"✓ Successfully generated {num_images} captcha images!")
    print(f"✓ Saved to: {os.path.abspath(output_dir)}")

def generate_single_captcha(text, output_dir=OUTPUT_FOLDER):
    """Generate a single captcha with specific text"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_captcha = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    image = image_captcha.generate_image(text)
    
    filename = f"{text}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    
    print(f"✓ Generated captcha: {filename}")
    print(f"✓ Saved to: {os.path.abspath(filepath)}")

if __name__ == "__main__":
    # Generate random captchas
    generate_captchas(num_images=NUM_CAPTCHAS)
    
    # Optionally generate specific captchas
    # generate_single_captcha("Hello123")
    # generate_single_captcha("Test456")