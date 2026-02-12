import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    """Enforces strict 128x128 resolution to satisfy 25,088 dense nodes."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.float32)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Strict Resize: This resolves the matrix mismatch
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Normalization
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Calculates Error Level Analysis to find pixel inconsistencies."""
    original = Image.open(image_file).convert('RGB')
    temp_p = 'temp_ela.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    
    # Calculate difference
    ela_image = ImageChops.difference(original, temporary)
    
    # Scale for visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)