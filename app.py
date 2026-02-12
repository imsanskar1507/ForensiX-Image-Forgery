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

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None

# --- DATABASE & UTILS (Original Logic Preserved) ---
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

# --- UI CSS (Your Design) ---
st.markdown("""<style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
</style>""", unsafe_allow_html=True)

if not st.session_state["logged_in"]:
    # Login logic remains exactly as you had it
    st.title("🛰️ ForensiX Login")
    with st.form("login"):
        u = st.text_input("AGENT ID")
        p = st.text_input("ACCESS KEY", type="password")
        if st.form_submit_button("AUTHORIZE"):
            if u == "sanskar": # Simple check for demo
                st.session_state["logged_in"] = True
                st.rerun()
else:
    # --- DASHBOARD ---
    st.markdown("## 🛰️ ForensiX Investigation Dashboard")
    
    @st.cache_resource
    def load_forensic_model():
        return load_model('forgery_detector.h5', compile=False)
    
    model = load_forensic_model()

    with st.sidebar:
        st.markdown(f"**⚡ OPERATIVE: {st.session_state['user'].upper()}**")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        # Side-by-Side Comparison (Restored)
        for f in files:
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f)
            with c_o: st.image(f, caption="SOURCE")
            with c_h: st.image(ela_img, caption="ELA ANALYSIS")

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            for f in files:
                tmp = f"temp_{f.name}"
                with open(tmp, "wb") as b: b.write(f.getbuffer())
                
                proc = prepare_image_for_cnn(tmp)
                input_tensor = np.expand_dims(proc, axis=0)
                
                try:
                    # Prediction logic
                    pred = model.predict(input_tensor)[0][0]
                    results.append({"FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"})
                except ValueError:
                    st.error(f"❌ Shape Mismatch: Your model expects a different size. Check if it was trained on 128x128.")
                    st.stop()
                
                os.remove(tmp)
            st.session_state["analysis_results"] = results
            st.table(pd.DataFrame(results))