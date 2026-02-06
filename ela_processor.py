from PIL import Image, ImageChops, ImageEnhance
import os

def convert_to_ela_image(path, quality=90):
    """
    Computes the Error Level Analysis of an image.
    """
    original = Image.open(path).convert('RGB')
    
    # 1. Save the image at a specific compression quality
    temp_filename = 'temp_resaved.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    resaved = Image.open(temp_filename)
    
    # 2. Calculate the absolute difference between original and resaved
    # Areas with high difference indicate potential forgery
    ela_image = ImageChops.difference(original, resaved)
    
    # 3. Amplify the difference so the CNN can see it clearly
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Remove the temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image

# --- Quick Test ---
# If you have an image, uncomment the lines below to see it work:
# test_img = convert_to_ela_image('path_to_your_image.jpg')
# test_img.show()