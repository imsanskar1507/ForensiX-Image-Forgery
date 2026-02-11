import streamlit as st
import numpy as np
import os
import cv2
import dlib
import hashlib
import sqlite3
import io
import zipfile
import pandas as pd
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from processor import prepare_image_for_cnn, convert_to_ela_image
from metadata_scanner import scan_metadata
from report_gen import create_pdf_report

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Load dlib models safely [cite: 139]
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

@st.cache_resource
def load_predictor():
    if os.path.exists(PREDICTOR_PATH):
        return dlib.shape_predictor(PREDICTOR_PATH)
    return None

predictor = load_predictor()

# --- RESEARCH METHODOLOGY LOGIC ---
def analyze_facial_landmarks(image_path):
    """Detects 68 landmarks to analyze eye, nose, and lip shapes[cite: 138, 140]."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    landmark_data = {"detected": False, "ear": 0.0, "status": "No Face Detected"}
    
    if predictor is None:
        landmark_data["status"] = "Predictor File Missing"
        return landmark_data

    for face in faces:
        landmark_data["detected"] = True
        landmarks = predictor(gray, face) [cite: 140]
        
        # Calculate EAR for blink detection [cite: 147, 150]
        def get_ear(eye_points):
            p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
            p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
            p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
            return (p2_p6 + p3_p5) / (2.0 * p1_p4) [cite: 151, 152]

        coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        left_eye = coords[36:42] [cite: 146]
        right_eye = coords[42:48]
        
        avg_ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0 [cite: 156]
        landmark_data["ear"] = round(avg_ear, 3)
        # Threshold 0.3 as used in the study [cite: 158]
        landmark_data["status"] = "Natural" if avg_ear > 0.3 else "Blink/Anomalous" [cite: 158]
        
    return landmark_data

# --- APP UI ---
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center;'>🛰️ ForensiX Image Forgery Detector</h1>", unsafe_allow_html=True)
    with st.form("Login"):
        u = st.text_input("AGENT ID")
        p = st.text_input("ACCESS KEY", type="password")
        if st.form_submit_button("AUTHORIZE"):
            if u == "sanskar":
                st.session_state["logged_in"] = True
                st.rerun()
else:
    # Main Dashboard
    st.markdown("## 🕵️ ForensiX Image Forgery Detector")
    
    if predictor is None:
        st.error(f"⚠️ Missing '{PREDICTOR_PATH}'. Facial landmark features are disabled.")

    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        results = []
        for f in files:
            tmp_path = f"temp_{f.name}"
            with open(tmp_path, "wb") as b: b.write(f.getbuffer())
            
            # ROI Cropping and Resizing to 224x224 [cite: 128, 129, 193]
            landmark_results = analyze_facial_landmarks(tmp_path)
            
            # Results Table [cite: 260]
            results.append({
                "EXHIBIT": f.name,
                "VERDICT": "SCANNING...",
                "EAR_SCORE": landmark_results["ear"],
                "FACE_LANDMARKS": landmark_results["status"]
            })
            
            st.image(f, caption=f"Processing {f.name}")
            os.remove(tmp_path)
            
        st.table(pd.DataFrame(results))