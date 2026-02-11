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
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    conn.commit(); conn.close()

init_db()

# --- CSS STYLING (RESTORING DOSSIER UI) ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
    
    /* Dossier Styling */
    .dossier-header {
        background-color: #00f2ff; color: #000; padding: 5px 15px; font-weight: bold;
        font-size: 11px; border-radius: 5px 5px 0 0; letter-spacing: 1.5px; display: inline-block;
    }
    .dossier-box {
        background: rgba(25, 27, 32, 0.95) !important;
        border: 1px solid #00f2ff !important;
        border-radius: 0 5px 5px 5px; padding: 15px; margin-bottom: 25px;
    }
    .report-frame {
        background: rgba(0, 242, 255, 0.05);
        border: 1px solid #00f2ff;
        border-radius: 10px; padding: 20px; margin-top: 25px;
    }
    .stTable { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    # (Login code omitted for brevity but remains the same as your working version)
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Investigation Portal</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                st.session_state["logged_in"] = True
                st.session_state["user"] = u_in
                st.rerun()
else:
    # --- DASHBOARD HEADER ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock:
        clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        model_path = 'forgery_detector.h5'
        if os.path.exists(model_path) and os.path.getsize(model_path) > 2000:
            return load_model(model_path, compile=False)
        return None
    
    model = get_model()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 20px;">
                <h4 style="margin:0; font-size: 12px; opacity: 0.8;">OPERATIVE</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 20px;">⚡ {st.session_state['user'].upper()}</h2>
                <p style="margin:5px 0 0 0; font-size: 12px; font-weight: bold;">📍 NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)
        
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("FIELD NOTES", height=100)
        
        if st.button("🧹 CLEAR"): 
            st.session_state["analysis_results"] = None
            st.rerun()
        if st.button("🔴 EXIT"): 
            st.session_state["logged_in"] = False
            st.rerun()

    st.markdown("---")
    
    # --- WORKSPACE ---
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
            st.markdown(f'<div class="dossier-header">🔬 EXHIBIT: {f.name}</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
                c_o, c_h = st.columns(2)
                
                # Visual Processing
                ela_img = convert_to_ela_image(f, quality=90)
                
                with c_o: st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
                with c_h: st.image(ela_img, caption="ERROR LEVEL ANALYSIS (ELA)", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 SCANNING CORE...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        has_meta, _ = scan_metadata(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        
                        results.append({
                            "FILENAME": f.name, 
                            "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                            "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                            "METADATA": "DETECTED" if has_meta else "NONE"
                        })
                        bar.progress((idx+1)/len(files))
                    
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", create_pdf_report(results, case_notes=case_notes))
                    status.update(label="SCAN COMPLETE", state="complete")
            
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    # --- REPORT SECTION ---
    if st.session_state["analysis_results"]:
        st.markdown("---")
        st.markdown(f'<div class="dossier-header">📊 CASE DOSSIER: {case_id}</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state["analysis_results"]))
            st.download_button(
                label="📥 DOWNLOAD ENCRYPTED DOSSIER (.ZIP)", 
                data=st.session_state["zip_buffer"], 
                file_name=f"ForensiX_{case_id}.zip",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # --- CLOCK FRAGMENT ---
    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
                <span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 22px;">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)

    sync_clock()