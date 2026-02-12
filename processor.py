import cv2
import numpy as np
from PIL import Image, ImageChops

def prepare_image_for_cnn(image_path):
    """Enforces 128x128 RGB resolution for 25,088 node compatibility."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.float32)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Generates Error Level Analysis to detect pixel inconsistencies."""
    original = Image.open(image_file).convert('RGB')
    temp_p = 'temp_ela.jpg'
    original.save(temp_p, 'JPEG', quality=quality)
    temporary = Image.open(temp_p)
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)