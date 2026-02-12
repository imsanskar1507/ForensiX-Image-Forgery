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
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata 
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session States (Preserving Nagpur Operative Profile for Sanskar Dhore)
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, "nagpur"))
    conn.commit(); conn.close()

init_db()

# --- UI CSS (Restored Design) ---
st.markdown("""<style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
    .dossier-box { background: rgba(25, 27, 32, 0.95); border: 1px solid #00f2ff; border-radius: 5px; padding: 10px; }
</style>""", unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login"):
            u = st.text_input("AGENT ID")
            p = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                if u == "sanskar":
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = "sanskar"
                    st.rerun()
else:
    # --- DASHBOARD ---
    @st.cache_resource
    def get_model():
        return load_model('forgery_detector.h5', compile=False)
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ OPERATIVE: {st.session_state['user'].upper()}**")
        st.markdown(f"**📍 NAGPUR_MS_IN**")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        # Side-by-Side UI (Restored)
        for f in files:
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f)
            with c_o: st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with c_h: st.image(ela_img, caption="ELA ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            bar = st.progress(0)
            for idx, f in enumerate(files):
                tmp = f"temp_{f.name}"
                with open(tmp, "wb") as b: b.write(f.getbuffer())
                
                # Pre-processing using 128x128
                proc = prepare_image_for_cnn(tmp)
                input_tensor = proc.reshape((1, 128, 128, 3)).astype('float32')
                
                # Predicting with hard-coded shape fix
                pred = model.predict(input_tensor, verbose=0)[0][0]
                os.remove(tmp)

                results.append({
                    "FILENAME": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%"
                })
                bar.progress((idx + 1) / len(files))
            
            st.session_state["analysis_results"] = results
            st.table(pd.DataFrame(results))

    @st.fragment(run_every="1s")
    def sync_clock():
        st.sidebar.markdown(f"**🕒 {datetime.now(IST).strftime('%I:%M:%S %p')}**")
    sync_clock()