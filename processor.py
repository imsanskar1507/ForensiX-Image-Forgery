import cv2
import numpy as np
from PIL import Image, ImageChops

# --- SAFE DLIB INITIALIZATION ---
try:
    import dlib
    detector = dlib.get_frontal_face_detector()
    # Path to the predictor file cited in your research paper
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
    """Calculates EAR for deepfake blink detection"""
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def prepare_image_for_cnn(image_path):
    """ROI Face Cropping & 224x224 RGB Resizing for Customized CNN"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if detector:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            f = faces[0]
            # ROI Cropping as per research methodology
            img = img[max(0, f.top()):f.bottom(), max(0, f.left()):f.right()]
    
    img = cv2.resize(img, (224, 224))
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Error Level Analysis for forgery detection"""
    original = Image.open(image_file).convert('RGB')
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    temporary = Image.open(temp_path)
    
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    
    # Cast scale to int to avoid TypeError
    scale = int(255.0 / max_diff)
    return Image.blend(ela_image, Image.new("RGB", ela_image.size, (0,0,0)), 1 - scale/255.0)