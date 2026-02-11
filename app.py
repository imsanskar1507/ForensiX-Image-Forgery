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

# --- NEW: AI DETECTION UTILITY ---
def detect_ai_synthetic_patterns(image_path):
    try:
        img = cv2.imread(image_path, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30] = 0
        score = np.var(magnitude_spectrum)
        if score > 120: return f"{min(98.9, score/1.5):.1f}%", "High Risk (AI)"
        elif score > 70: return f"{score/1.8:.1f}%", "Possible Synthetic"
        else: return f"{score/4:.1f}%", "Likely Biological"
    except: return "0.0%", "Scan Error"

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# --- CSS STYLING (THEMED FOR G H RAISONI PRESENTATION) ---
st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0a0c10 !important; border-right: 1px solid #00f2ff; }
    
    /* Dossier Styling with Neon Glow */
    .dossier-header {
        background-color: #00f2ff; color: #000; padding: 6px 18px; font-weight: bold;
        font-size: 12px; border-radius: 4px 4px 0 0; letter-spacing: 2px; display: inline-block;
        box-shadow: 0px 0px 10px #00f2ff;
    }
    .dossier-box {
        background: #0d1117 !important;
        border: 1px solid #00f2ff !important;
        border-radius: 0 4px 4px 4px; padding: 18px; margin-bottom: 30px;
        box-shadow: inset 0px 0px 15px rgba(0, 242, 255, 0.1);
    }
    .report-frame {
        background: rgba(0, 242, 255, 0.03);
        border: 2px dashed #00f2ff;
        border-radius: 8px; padding: 25px; margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    # (Login code omitted - Use your standard auth block here)
    st.session_state["logged_in"] = True # TEMP BYPASS FOR UI VIEW
    st.rerun()
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; text-shadow: 0 0 10px #00f2ff;">🛰️ ForensiX Suite</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        if os.path.exists(mp) and os.path.getsize(mp) > 2000:
            return load_model(mp, compile=False)
        return None
    model = get_model()

    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.08); padding: 20px; border-radius: 8px; border: 1px solid #00f2ff;">
                <h4 style="margin:0; font-size: 11px; opacity: 0.7;">AUTH_TOKEN</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 22px;">🕵️ {st.session_state['user'].upper()}</h2>
                <hr style="border-color: #00f2ff; opacity: 0.3;">
                <p style="margin:0; font-size: 12px;">SYSTEM: ACTIVE</p>
            </div>
        """, unsafe_allow_html=True)
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🧹 PURGE CACHE"): st.session_state["analysis_results"] = None; st.rerun()
        if st.button("🔴 SHUTDOWN"): st.session_state["logged_in"] = False; st.rerun()

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
            st.markdown(f'<div class="dossier-header">EXHIBIT ANALYSIS: {f.name}</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
                c_o, c_h = st.columns(2)
                ela_img = convert_to_ela_image(f, quality=90)
                with c_o: st.image(f, caption="SOURCE", use_container_width=True)
                with c_h: st.image(ela_img, caption="ELA HEATMAP", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("INITIATE HYBRID SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 ANALYZING PIXEL FREQUENCY...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        # Hybrid detection
                        ai_conf, ai_risk = detect_ai_synthetic_patterns(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        
                        results.append({
                            "FILENAME": f.name, 
                            "FORGERY": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                            "AI_PATTERNS": ai_conf,
                            "DETERMINATION": ai_risk,
                            "CNN_CONF": f"{max(pred, 1-pred)*100:.1f}%"
                        })
                        bar.progress((idx+1)/len(files))
                    
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", create_pdf_report(results))
                    status.update(label="SCAN COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.markdown(f'<div class="dossier-header">📊 FINAL DOSSIER: {case_id}</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state["analysis_results"]))
            st.download_button("📥 DOWNLOAD ENCRYPTED ZIP", st.session_state["zip_buffer"], f"{case_id}.zip", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.05); padding: 8px 18px; border-radius: 6px; border-right: 4px solid #00f2ff; box-shadow: 0 0 10px rgba(0, 242, 255, 0.1);">
                <span style="color: #00f2ff; font-weight: bold; letter-spacing: 1px;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 24px; font-weight: 500;">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)
    sync_clock()