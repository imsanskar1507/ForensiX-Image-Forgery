import cv2
import numpy as np
import os
from PIL import Image, ImageChops
import io

def prepare_image_for_cnn(image_path):
    """Strictly enforces 128x128 resolution."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Generates ELA by comparing original with a re-compressed version."""
    # Create a copy of the file in memory to avoid pointer errors
    img_bytes = image_file.getvalue()
    original = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Process ELA
    temp_p = 'temp_ela_process.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    
    ela_image = ImageChops.difference(original, temporary)
    
    # Scale for visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    
    result = Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)
    
    if os.path.exists(temp_p):
        os.remove(temp_p)
        
    return result