import numpy as np
from PIL import Image, ImageChops
import os

def convert_to_ela_image(image_file, quality=90):
    """
    Performs Error Level Analysis (ELA) by calculating the pixel-wise difference 
    between the original and a resaved version[cite: 121].
    """
    temp_filename = 'temp_ela.jpg'
    ela_filename = 'ela_result.jpg'
    
    # Open original image
    image = Image.open(image_file).convert('RGB')
    
    # Resave at specific quality to detect compression inconsistencies [cite: 121]
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    # Calculate pixel difference (ELA)
    ela_image = ImageChops.difference(image, temp_image)
    
    # Scale intensities to make the difference visible in the UI [cite: 12, 186]
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageChops.constant(ela_image, scale)
    
    # Cleanup temp files
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image

def prepare_image_for_cnn(image_path):
    """
    Standardizes image to 224x224 RGB and normalizes pixel values 
    to 0-1 range for the 20-layer CNN[cite: 115, 116, 125].
    """
    # Resize to the strict requirement mentioned in your paper [cite: 115]
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224)) 
    
    # Convert to array and normalize (0 to 1) [cite: 119]
    image_array = np.array(image).astype('float32') / 255.0
    
    return image_array