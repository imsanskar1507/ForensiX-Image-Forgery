import streamlit as st
import numpy as np
import os
import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
import pytz
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session State Initialization
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user" not in st.session_state: st.session_state.user = "Unknown"

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('forensix.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    # Default operative: sanskar
    hp = hashlib.sha256("detective2026".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users VALUES ('sanskar', ?)", (hp,))
    conn.commit()
    conn.close()

init_db()

# --- CUSTOM UI CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
    .stButton>button { background-color: transparent; color: #00f2ff; border: 1px solid #00f2ff; width: 100%; }
    .stButton>button:hover { background-color: #00f2ff; color: #000; }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        u = st.text_input("AGENT ID")
        p = st.text_input("ACCESS KEY", type="password")
        if st.button("AUTHORIZE SESSION"):
            if u == "sanskar" and p == "detective2026":
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid Credentials")
else:
    # --- DASHBOARD UI ---
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="color:#00f2ff;">🛰️ Forensic Investigation Dashboard</h2>', unsafe_allow_html=True)
    with col_clock: 
        now = datetime.now(IST).strftime('%I:%M:%S %p')
        st.markdown(f"<p style='text-align:right;'>🕒 {now}</p>", unsafe_allow_html=True)

    # SIDEBAR: Operative Profile & Case Management
    with st.sidebar:
        st.markdown(f"""<div style="background: rgba(0, 242, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #00f2ff;">
            <h4 style="margin:0; font-size: 12px; opacity: 0.7;">OPERATIVE</h4>
            <h2 style="margin:0; color: #00f2ff;">⚡ {st.session_state.user.upper()}</h2>
            <p style="margin:5px 0 0 0; font-size: 12px; font-weight: bold;">📍 NAGPUR_MS_IN</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("FIELD NOTES", height=100)
        
        if st.button("🔴 EXIT SESSION"):
            st.session_state.logged_in = False
            st.rerun()

    # MAIN CONTENT
    @st.cache_resource
    def load_engine():
        return load_model('forgery_detector.h5', compile=False) if os.path.exists('forgery_detector.h5') else None
    
    model = load_engine()

    files = st.file_uploader("UPLOAD EVIDENCE EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        for f in files:
            st.markdown(f"<div style='border-left: 3px solid #00f2ff; padding-left: 10px; margin: 20px 0;'>EXHIBIT: {f.name}</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with col2: st.image(convert_to_ela_image(f), caption="ELA HEATMAP", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            bar = st.progress(0)
            for idx, f in enumerate(files):
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b: b.write(f.getbuffer())
                
                # Pre-processing (128x128 Fix)
                img_data = prepare_image_for_cnn(t_path)
                tensor = np.expand_dims(img_data, axis=0)
                
                # Prediction
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_path)
                
                results.append({
                    "EXHIBIT": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%"
                })
                bar.progress((idx + 1) / len(files))
            
            st.markdown("### 📊 FINAL DETERMINATION REPORT")
            st.table(pd.DataFrame(results))