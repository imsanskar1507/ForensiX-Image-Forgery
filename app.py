import streamlit as st
import numpy as np
import os
import pandas as pd
import hashlib
from datetime import datetime
import pytz
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

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
    st.markdown("<h1 style='text-align:center;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        u = st.text_input("AGENT ID")
        p = st.text_input("ACCESS KEY", type="password")
        if st.button("AUTHORIZE SESSION"):
            if u == "sanskar" and p == "detective2026":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
else:
    # --- DASHBOARD ---
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="color:#00f2ff;">🛰️ Forensic Investigation Dashboard</h2>', unsafe_allow_html=True)
    with col_clock: 
        now = datetime.now(IST).strftime('%I:%M:%S %p')
        st.markdown(f"<p style='text-align:right;'>🕒 {now}</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""<div style="background: rgba(0, 242, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #00f2ff;">
            <h4 style="margin:0; font-size: 12px; opacity: 0.7;">OPERATIVE</h4>
            <h2 style="margin:0; color: #00f2ff;">⚡ SANSKAR</h2>
            <p style="margin:5px 0 0 0; font-size: 12px; font-weight: bold;">📍 NAGPUR_MS_IN</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🔴 EXIT SESSION"):
            st.session_state.logged_in = False
            st.rerun()

    @st.cache_resource
    def load_engine():
        return load_model('forgery_detector.h5', compile=False) if os.path.exists('forgery_detector.h5') else None
    
    model = load_engine()

    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        for f in files:
            st.markdown(f"**EXHIBIT: {f.name}**")
            
            # 1. Reset pointer for display
            f.seek(0)
            
            # 2. Get ELA Image
            ela_img = convert_to_ela_image(f)
            
            # 3. Display
            col1, col2 = st.columns(2)
            with col1: 
                f.seek(0) # Ensure pointer is ready for st.image
                st.image(f, caption="SOURCE", use_container_width=True)
            with col2: 
                st.image(ela_img, caption="ELA HEATMAP", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            for f in files:
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b: 
                    f.seek(0) # Ensure full file is written to temp
                    b.write(f.read())
                
                img_data = prepare_image_for_cnn(t_path)
                tensor = np.expand_dims(img_data, axis=0)
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_path)
                
                results.append({"FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"})
            st.table(pd.DataFrame(results))