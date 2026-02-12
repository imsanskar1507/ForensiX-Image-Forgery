import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    """Enforces 128x128 resolution to match the 25,088 node requirement."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Strict resize to prevent ValueError during model.predict
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Generates the ELA Heatmap by highlighting compression differences."""
    # Reset pointer so the processor sees the full data stream
    image_file.seek(0)
    original = Image.open(image_file).convert('RGB')
    
    temp_p = 'temp_ela_work.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    
    # Calculate difference between original and re-compressed version
    ela_image = ImageChops.difference(original, temporary)
    
    # Scale for high-contrast visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    
    result = Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)
    
    if os.path.exists(temp_p):
        os.remove(temp_p)
        
    return result