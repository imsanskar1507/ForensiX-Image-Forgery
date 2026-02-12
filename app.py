import streamlit as st
import numpy as np
import os
import pandas as pd
import sqlite3
import hashlib
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- UI CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide", page_icon="🕵️")

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    # Default operative: sanskar
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES ('sanskar', ?)", (hp,))
    conn.commit()
    conn.close()

init_db()

# --- AUTH LOGIC ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

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
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Credentials")
else:
    # --- DASHBOARD ---
    st.sidebar.markdown(f"**⚡ OPERATIVE: SANSKAR**")
    st.sidebar.markdown("**📍 LOCATION: NAGPUR_MS_IN**")
    
    @st.cache_resource
    def load_engine():
        # compile=False speeds up loading and prevents local errors
        return load_model('forgery_detector.h5', compile=False)
    
    model = load_engine()

    st.markdown("## 🛰️ Forensic Evidence Analysis")
    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        for f in files:
            col1, col2 = st.columns(2)
            with col1: 
                st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with col2: 
                st.image(convert_to_ela_image(f), caption="ELA ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            bar = st.progress(0)
            for idx, f in enumerate(files):
                # Temporary file for processing
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b:
                    b.write(f.getbuffer())
                
                # 1. Pre-process (Strict 128x128)
                img_data = prepare_image_for_cnn(t_path)
                
                # 2. Add Batch Dimension (The Fix for ValueError)
                # tensor shape becomes (1, 128, 128, 3)
                tensor = np.expand_dims(img_data, axis=0)
                
                # 3. Model Prediction
                pred = model.predict(tensor, verbose=0)[0][0]
                
                os.remove(t_path)
                results.append({
                    "FILENAME": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "PROBABILITY": f"{pred:.4f}"
                })
                bar.progress((idx + 1) / len(files))
            
            st.markdown("### 📊 FINAL DETERMINATION REPORT")
            st.table(pd.DataFrame(results))

    if st.sidebar.button("🔴 EXIT SESSION"):
        st.session_state["logged_in"] = False
        st.rerun()