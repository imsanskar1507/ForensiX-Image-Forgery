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

# Session State Initialization
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state: st.session_state["case_log"] = []
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        # Operative: Sanskar Dhore
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit()
    conn.close()

def check_user(u, p):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hp = hashlib.sha256(p.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hp))
    res = c.fetchone()
    conn.close()
    return res

init_db()

# --- CORE UTILITIES ---
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def generate_heatmap(original_img_bytes, ela_img):
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width, _ = original.shape
    ela_cv = np.array(ela_img.convert('RGB'))
    gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)
    heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_color, (width, height))
    return cv2.addWeighted(original, 0.6, heatmap_resized, 0.4, 0)

# --- APP LOGIC ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Image Forgery Detector</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div style="background:rgba(15,17,22,0.75); border:2px solid #00f2ff; border-radius:15px; padding:25px;">', unsafe_allow_html=True)
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                if check_user(u_in, p_in):
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    st.rerun()
                else: st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # --- DASHBOARD ---
    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        if os.path.exists(mp): return load_model(mp)
        return None
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ OPERATIVE: {st.session_state['user'].upper()}**")
        st.markdown(f"**📍 LOCATION: NAGPUR_MS_IN**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
            f_bytes = f.getvalue()
            st.info(f"🧬 EXHIBIT {f.name} | HASH: {get_file_hash(f_bytes)}")
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f)
            heat_img = generate_heatmap(f_bytes, ela_img)
            with c_o: st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with c_h: st.image(heat_img, caption="HEATMAP ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            for f in files:
                tmp = f"temp_{f.name}"
                with open(tmp, "wb") as b: b.write(f.getbuffer())
                proc = prepare_image_for_cnn(tmp)
                # Original expand_dims logic
                pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                os.remove(tmp)
                results.append({"FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"})
            
            st.session_state["analysis_results"] = results
            st.table(pd.DataFrame(results))