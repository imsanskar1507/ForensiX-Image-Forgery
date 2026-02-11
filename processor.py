import cv2
import numpy as np
from PIL import Image, ImageChops

# --- SAFE DLIB IMPORT ---
try:
    import dlib
    detector = dlib.get_frontal_face_detector() [cite: 139]
    # Path to the predictor file described in your research [cite: 141]
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
except ImportError:
    dlib = None
    detector = None
    predictor = None

def get_ear(eye_points):
    """Calculates EAR using the research paper formula [cite: 151, 152]"""
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def prepare_image_for_cnn(image_path):
    """Standardizes input to 224x224 RGB as per customized CNN methodology [cite: 129, 193]"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) [cite: 129]
    
    # ROI Face Cropping [cite: 179]
    if detector:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            f = faces[0]
            img = img[max(0, f.top()):f.bottom(), max(0, f.left()):f.right()]
    
    img = cv2.resize(img, (224, 224)) [cite: 193]
    return img.astype('float32') / 255.0

def convert_to_ela_image(image_file, quality=90):
    """Error Level Analysis for detecting digital manipulation [cite: 88]"""
    original = Image.open(image_file).convert('RGB')
    temp_path = "temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    temporary = Image.open(temp_path)
    
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    return ImageChops.constant(ela_image, scale)