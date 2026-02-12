import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    """
    Strictly enforces the 224x224 RGB input required by the Customized CNN.
    Ensures the array is float32 and normalized to [0, 1].
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.float32)

    # Convert BGR (OpenCV default) to RGB (Model requirement)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Strict resize to 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    return img

# Update these functions in your processor.py
def convert_to_ela_image(image_file, quality=90):
    original = Image.open(image_file).convert('RGB')
    temp_p = 'temp_ela.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    # CRITICAL: Scale must be an integer for PIL.Image.new
    scale = int(255.0 / max_diff)
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)

def prepare_image_for_cnn(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure strict 224x224 shape to avoid ValueError
    img = cv2.resize(img, (224, 224))
    return img.astype('float32') / 255.0