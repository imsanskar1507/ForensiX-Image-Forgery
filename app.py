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
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")

# LOCK TIME TO INDIA STANDARD TIME (IST)
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state:
    st.session_state["case_log"] = []
# New states to handle persistent report display
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None
if "zip_report" not in st.session_state:
    st.session_state["zip_report"] = None

# (All your core utility functions: get_timestamp, log_forensic_action, etc. remain the same)
# ...

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit()
    conn.close()

init_db()

# (Your existing CSS styling block remains the same)
# ...

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    # (Your existing Login/Register/Forgot UI logic remains the same)
    # ...
    pass
else:
    # --- DASHBOARD HEADER ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock:
        clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = os.path.join(os.path.dirname(__file__), 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None
    
    model = get_model()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 25px;">
                <h4 style="margin:0; font-size: 14px; opacity: 0.8;">OPERATIVE STATUS</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 22px;">⚡ {st.session_state['user'].upper()}</h2>
                <p style="margin:10px 0 0 0; font-size: 14px; color: #00f2ff; font-weight: bold;">📍 LOCATION: NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)
        
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        st.markdown('<div class="dossier-header">📝 INVESTIGATION LOG</div><div class="dossier-box">', unsafe_allow_html=True)
        case_notes = st.text_area("FIELD NOTES", height=150, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🧹 CLEAR CASE"):
            st.session_state["scan_results"] = None
            st.session_state["zip_report"] = None
            st.rerun()

    # --- ANALYSIS WORKSPACE ---
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        if st.button("INITIATE DEEP SCAN"):
            results = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 SCANNING CORE...", expanded=True) as status:
                    for idx, f in enumerate(files):
                        # Temp save for processing
                        tmp_name = f"tmp_{f.name}"
                        with open(tmp_name, "wb") as b: b.write(f.getbuffer())
                        
                        # Metadata and CNN Analysis
                        has_meta, _ = scan_metadata(tmp_name)
                        processed = prepare_image_for_cnn(tmp_name)
                        prediction = model.predict(np.expand_dims(processed, axis=0))[0][0]
                        os.remove(tmp_name)

                        # Verdict Logic
                        verdict = "🚩 FORGERY" if prediction > 0.5 else "🏳️ CLEAN"
                        confidence = f"{max(prediction, 1-prediction)*100:.2f}%"
                        
                        results.append({
                            "FILENAME": f.name,
                            "VERDICT": verdict,
                            "CONFIDENCE": confidence,
                            "METADATA": "DETECTED" if has_meta else "NONE"
                        })
                        bar.progress((idx+1)/len(files))
                    
                    # Generate PDF and Zip
                    pdf_bytes = create_pdf_report(results, case_notes=case_notes)
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", pdf_bytes)
                    status.update(label="ANALYSIS COMPLETE", state="complete")
            
            st.session_state["scan_results"] = results
            st.session_state["zip_report"] = zip_buffer.getvalue()

    # --- DYNAMIC REPORT SECTION ---
    if st.session_state["scan_results"]:
        st.markdown("---")
        st.markdown(f'<div class="dossier-header">📊 FINAL ANALYSIS REPORT: {case_id}</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
            
            # Summary Table
            df = pd.DataFrame(st.session_state["scan_results"])
            st.table(df)
            
            # Download Action
            st.download_button(
                label="📥 DOWNLOAD ENCRYPTED CASE DOSSIER (.ZIP)",
                data=st.session_state["zip_report"],
                file_name=f"ForensiX_{case_id}_Report.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # (Your existing Clock loop logic remains the same)
    # ...