import numpy as np
from PIL import Image, ImageChops
import os

def convert_to_ela_image(image_file, quality=90):
    """
    Performs Error Level Analysis (ELA) to identify compression inconsistencies[cite: 121].
    """
    image = Image.open(image_file).convert('RGB')
    temp_filename = 'temp_ela.jpg'
    
    # Resave at known quality to detect mathematical anomalies [cite: 121]
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    # Calculate pixel difference [cite: 121]
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    return ImageChops.constant(ela_image, scale)

def prepare_image_for_cnn(image_path):
    """
    Standardizes image to 224x224 RGB and normalizes pixels for the 20-layer CNN[cite: 115, 116].
    """
    image = Image.open(image_path).convert('RGB')
    # Resizing to exact research specifications [cite: 115]
    image = image.resize((224, 224)) 
    
    # Normalizing pixel values to 0-1 range to prevent saturated verdicts 
    image_array = np.array(image).astype('float32') / 255.0
    
    return image_array