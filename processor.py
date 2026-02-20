import numpy as np
from PIL import Image, ImageChops
import os

def convert_to_ela_image(image_file, quality=90):
    image = Image.open(image_file).convert('RGB')
    temp_filename = 'temp_ela.jpg'
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    return ImageChops.constant(ela_image, scale)

def prepare_image_for_cnn(image_path):
    # Resize to 224x224 as per research paper specs [cite: 115]
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224)) 
    
    # CRITICAL: Normalize pixels to range [0, 1] 
    image_array = np.array(image).astype('float32') / 255.0
    
    return image_array