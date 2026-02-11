import cv2
import numpy as np
import dlib
from PIL import Image, ImageChops

# --- INITIALIZE DLIB ---
# This uses the 68-landmark predictor described in your research [cite: 138, 140]
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    predictor = None

def get_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) using the research formula[cite: 151, 152].
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical distances
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

def analyze_temporal_features(frame):
    """
    Detects 68 facial landmarks and calculates EAR to identify 
    unnatural blinking patterns common in deepfakes[cite: 143, 144].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, "No Face Detected"

    for face in faces:
        landmarks = predictor(gray, face)
        coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        # Eye landmarks: points 37–46 [cite: 146]
        left_eye = coords[36:42]
        right_eye = coords[42:48]
        
        ear_left = get_ear(left_eye)
        ear_right = get_ear(right_eye)
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Threshold 0.3 as specified in the study [cite: 158]
        status = "Blinking/Closed" if avg_ear < 0.3 else "Open"
        return avg_ear, status

def prepare_image_for_cnn(image_path):
    """
    Standardizes input images to 224x224 RGB as per the 
    Customized CNN requirements[cite: 129, 193].
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ROI Face Cropping logic [cite: 128, 179]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        # Use the first detected face ROI
        x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        # Padding to avoid losing features
        img = img[max(0, y):y+h, max(0, x):x+w]
        
    # Resize to 224x224 [cite: 193]
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return img

def convert_to_ela_image(image_file, quality=90):
    """
    Performs Error Level Analysis to detect compression 
    mismatches in forged images[cite: 88].
    """
    original = Image.open(image_file).convert('RGB')
    
    # Temporary save with reduced quality
    temp_filename = 'temp_ela.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    temporary = Image.open(temp_filename)
    
    # Calculate difference between original and compressed [cite: 101]
    ela_image = ImageChops.difference(original, temporary)
    
    # Scale intensities for visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageChops.constant(ela_image, scale)
    
    return ela_image