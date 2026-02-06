from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np

def convert_to_ela_image(path, quality=90):
    """
    This function detects forgery by resaving the image and 
    finding the difference in compression levels.
    """
    # Open the original image
    original = Image.open(path).convert('RGB')
    
    # Save it temporarily at a specific compression quality
    temp_filename = 'temp_resaved.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    resaved = Image.open(temp_filename)
    
    # Calculate the pixel-by-pixel difference
    ela_image = ImageChops.difference(original, resaved)
    
    # Amplify the difference so the CNN can 'see' the anomalies better
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Clean up the temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image

def prepare_image_for_cnn(image_path, size=(128, 128)):
    """
    Converts to ELA, resizes, and turns into a NumPy array for the model.
    """
    ela_img = convert_to_ela_image(image_path)
    ela_img = ela_img.resize(size)
    return np.array(ela_img) / 255.0  # Normalization (0 to 1)