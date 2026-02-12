import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pytz 
import pandas as pd
import sqlite3
import hashlib
import cv2
import io
import zipfile
import time 
import matplotlib.pyplot as plt
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata 
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session State Initialization (Maintained as per user request)
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state: st.session_state["case_log"] = []
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# ... (Core Utilities, Forgery Type Classifier, and Database Engine kept exactly as provided) ...

init_db()

# --- APP LOGIC ---
if not st.session_state["logged_in"]:
    # ... (Login UI Logic kept exactly as provided) ...
    pass
else:
    # --- DASHBOARD LOGIC ---
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Image Forgery Detector</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        if os.path.exists(mp) and os.path.getsize(mp) > 2000:
            return load_model(mp, compile=False) 
        return None
    model = get_model()

    # ... (Sidebar logic kept exactly as provided) ...

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        # Display Preview Section (Kept as provided)
        for f in files:
            # Display logic...
            pass

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 SCANNING PIXELS...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        # 1. METADATA & ELA (As per provided logic)
                        has_meta, meta_info = scan_metadata(tmp)
                        software_tag = "🏳️ ORIGINAL"
                        if has_meta: software_tag = "🚩 EDITED"
                        
                        ela_img_data = convert_to_ela_image(f, quality=90)
                        forgery_type = classify_forgery_type(ela_img_data)
                        
                        # 2. CNN PREDICTION (THE FIX)
                        proc = prepare_image_for_cnn(tmp)
                        
                        # FORCE SHAPE: Ensure image is exactly (224, 224, 3)
                        if proc.shape[:2] != (224, 224):
                            proc = cv2.resize(proc, (224, 224))
                        
                        # Ensure it's 4D: (1, 224, 224, 3)
                        input_tensor = proc.reshape((1, 224, 224, 3)).astype('float32')
                        
                        # Predict
                        pred_raw = model.predict(input_tensor)
                        pred = pred_raw[0][0]
                        
                        os.remove(tmp)

                        results.append({
                            "FILENAME": f.name, 
                            "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                            "FORGERY TYPE": forgery_type,
                            "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                            "METADATA": software_tag
                        })
                        bar.progress((idx+1)/len(files))
                    
                    zf.writestr(f"Forensic_Report.pdf", create_pdf_report(results))
                    status.update(label="COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    # ... (Final Report Table and Clock logic kept exactly as provided) ...