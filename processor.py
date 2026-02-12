import cv2
import numpy as np
import os
from PIL import Image, ImageChops

# --- DLIB INITIALIZATION (For Deepfake/Facial Analysis) ---
try:
    import dlib
    detector = dlib.get_frontal_face_detector()
    # The 68-landmark predictor file should be in your root directory
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    if os.path.exists(PREDICTOR_PATH):
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    else:
        predictor = None
except ImportError:
    dlib = None
    detector = None
    predictor = None

def get_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) for deepfake detection.
    Used to identify unnatural blinking patterns.
    """
    # Vertical distances
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def prepare_image_for_cnn(image_path):
    """
    Pre-processing pipeline for the 20-layer Customized CNN.
    1. Reads image and converts to RGB.
    2. Performs ROI Face Cropping (optional based on detection).
    3. Resizes strictly to 224x224 (prevents ValueError).
    4. Normalizes pixel values.
    """
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ROI Face Cropping logic using dlib
    if detector:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            f = faces[0]
            # Crop to face boundary
            img = img[max(0, f.top()):f.bottom(), max(0, f.left()):f.right()]
    
    # Strict resizing to match model input shape (224, 224, 3)
    img = cv2.resize(img, (224, 224))
    
    # Normalization [0, 1]
    img = img.astype('float32') / 255.0
    return img

def convert_to_ela_image(image_file, quality=90):
    """
    Error Level Analysis (ELA) for detecting digital manipulation.
    Calculates compression differences to highlight forged areas.
    """
    original = Image.open(image_file).convert('RGB')
    
    # Save a temporary copy with specified quality
    temp_path = 'temp_ela.jpg'
    original.save(temp_path, 'JPEG', quality=quality)
    temporary = Image.open(temp_path)
    
    # Get the absolute difference
    ela_image = ImageChops.difference(original, temporary)
    
    # Determine the scale for visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    # FIX: Cast to int to prevent TypeError in PIL functions
    scale = int(255.0 / max_diff)
    
    # Create the result by blending (more stable than ImageChops.constant)
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)