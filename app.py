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

# Initialize dlib for Facial Landmark Detection [cite: 139, 140]
detector = dlib.get_frontal_face_detector()
# Note: You must have 'shape_predictor_68_face_landmarks.dat' in your directory
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None

# --- TEMPORAL FEATURE ANALYSIS (RESEARCH METHODOLOGY) ---
def analyze_facial_landmarks(image_path):
    """Detects 68 landmarks to analyze eye, nose, and lip shapes[cite: 138, 164, 173]."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    landmark_data = {"detected": False, "ear": 0.0, "status": "No Face Detected"}
    
    for face in faces:
        landmark_data["detected"] = True
        landmarks = predictor(gray, face)
        
        # Calculate Eye Aspect Ratio (EAR) for blink detection [cite: 150, 151]
        def get_ear(eye_points):
            p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
            p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
            p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
            return (p2_p6 + p3_p5) / (2.0 * p1_p4)

        coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        left_eye = coords[36:42]
        right_eye = coords[42:48]
        
        avg_ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
        landmark_data["ear"] = round(avg_ear, 3)
        landmark_data["status"] = "Natural" if avg_ear > 0.2 else "Blink/Anomalous" [cite: 158]
        
    return landmark_data

# --- CUSTOMIZED CNN LOADER ---
@st.cache_resource
def load_custom_cnn():
    """Loads the 20-layer customized CNN architecture[cite: 209]."""
    model_path = 'forgery_detector.h5'
    if os.path.exists(model_path):
        # compile=False allows us to manually use Log-Loss metrics [cite: 303]
        return load_model(model_path, compile=False)
    return None

# --- UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    h1 { font-size: 55px !important; text-align: center; text-shadow: 0 0 20px #00f2ff; }
    .dossier-box { background: rgba(25, 27, 32, 0.95); border: 1px solid #00f2ff; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- MAIN APP ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1>🛰️ ForensiX Image Forgery Detector</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login"):
            u = st.text_input("AGENT ID")
            p = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE"):
                if u == "sanskar": # Linked to your user profile
                    st.session_state["logged_in"], st.session_state["user"] = True, u
                    st.rerun()
else:
    st.markdown("## 🛰️ ForensiX Image Forgery Detector")
    model = load_custom_cnn()
    
    with st.sidebar:
        st.info(f"AGENT: {st.session_state['user'].upper()}")
        case_id = st.text_input("CASE ID", "REF-RAISONI-2026")
        if st.button("🔴 LOGOUT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        results = []
        for f in files:
            # 1. Pre-processing: Resize to 224x224 RGB [cite: 129, 193]
            tmp_path = f"temp_{f.name}"
            with open(tmp_path, "wb") as b: b.write(f.getbuffer())
            
            # 2. Facial Landmark Analysis [cite: 125, 142]
            landmark_results = analyze_facial_landmarks(tmp_path)
            
            # 3. CNN Classification [cite: 132, 201]
            proc_img = prepare_image_for_cnn(tmp_path) # Standardized to 224x224
            prediction = model.predict(np.expand_dims(proc_img, axis=0))[0][0]
            
            # 4. Results Aggregation
            verdict = "🚩 FAKE" if prediction > 0.5 else "🏳️ REAL"
            results.append({
                "EXHIBIT": f.name,
                "VERDICT": verdict,
                "CNN_CONF": f"{max(prediction, 1-prediction)*100:.2f}%",
                "EAR_SCORE": landmark_results["ear"],
                "FACE_LANDMARKS": landmark_results["status"]
            })
            
            # Display View
            c1, c2 = st.columns(2)
            with c1: st.image(f, caption="SOURCE")
            with c2: st.image(convert_to_ela_image(f), caption="ELA ANALYSIS [cite: 88]")
            
            os.remove(tmp_path)

        if st.button("GENERATE FORENSIC DOSSIER"):
            st.session_state["analysis_results"] = results
            st.success("Analysis Complete based on G.H. Raisoni Methodology[cite: 6].")

    if st.session_state["analysis_results"]:
        st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state["analysis_results"])
        st.table(df)
        
        # Download results as Zip (Case Dossier)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download PDF/CSV Report", csv, f"{case_id}.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)