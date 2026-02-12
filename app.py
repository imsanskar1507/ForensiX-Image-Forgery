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
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session State Initialization (Operative: Sanskar Dhore)
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

def check_user(u, p):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hp = hashlib.sha256(p.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hp))
    res = c.fetchone(); conn.close()
    return res

init_db()

# --- CORE UTILITIES ---
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

# --- UI LOGIC ---
if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        with st.form("login"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                if check_user(u_in, p_in):
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    st.rerun()
                else: st.error("Invalid Credentials")
else:
    # --- DASHBOARD ---
    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        return load_model(mp, compile=False) if os.path.exists(mp) else None
    
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ OPERATIVE: {st.session_state['user'].upper()}**")
        st.markdown(f"**📍 LOCATION: NAGPUR_MS_IN**")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f)
            heat_img = generate_heatmap(f.getvalue(), ela_img)
            with c_o: st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with c_h: st.image(heat_img, caption="HEATMAP ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            bar = st.progress(0)
            for idx, f in enumerate(files):
                tmp = f"temp_{f.name}"
                with open(tmp, "wb") as b: b.write(f.getbuffer())
                
                # Pre-processing using 128x128 (Resolves 25088 ValueError)
                proc = prepare_image_for_cnn(tmp)
                input_tensor = proc.reshape((1, 128, 128, 3)).astype('float32')
                
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