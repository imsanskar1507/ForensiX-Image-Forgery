import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    """Enforces 128x128 resolution for matrix alignment."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Generates a visible ELA Heatmap by amplifying pixel differences."""
    image_file.seek(0)
    original = Image.open(image_file).convert('RGB')
    
    # Save a temporary re-compressed version
    temp_p = 'temp_ela_analysis.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    
    # Calculate the mathematical difference
    ela_image = ImageChops.difference(original, temporary)
    
    # AMPLIFICATION LOGIC:
    # We find the maximum difference and scale it to the 0-255 range.
    # If the image looks black, it's because max_diff is very small.
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    
    # We use a higher multiplier if the difference is very subtle
    scale = 255.0 / max_diff
    
    # Blend the difference onto a black background
    result = Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)
    
    if os.path.exists(temp_p):
        os.remove(temp_p)
        
    return result