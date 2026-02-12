import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure 224x224 to prevent matrix alignment issues
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA) 
    return img.astype('float32') / 255.0
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # HARD RESIZE: Enforce 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    return img

def convert_to_ela_image(image_file, quality=90):
    """ELA logic with integer casting fix for Pillow."""
    original = Image.open(image_file).convert('RGB')
    temp_p = 'temp_ela.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    
    # FIX: Scale must be an integer to avoid TypeError in PIL
    scale = int(255.0 / max_diff)
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)