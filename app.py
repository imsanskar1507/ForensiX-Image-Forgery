import streamlit as st
import numpy as np
import os
import pandas as pd
import hashlib
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide")

# Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- AUTHENTICATION UI ---
if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        with st.form("login_form"):
            st.markdown("### ENTER AGENT CREDENTIALS")
            u_input = st.text_input("AGENT ID (Case Sensitive)")
            p_input = st.text_input("ACCESS KEY", type="password")
            submit = st.form_submit_button("AUTHORIZE SESSION", use_container_width=True)
            
            if submit:
                # Direct check to ensure no database delay issues
                if u_input == "sanskar" and p_input == "detective2026":
                    st.session_state["logged_in"] = True
                    st.success("Access Granted. Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid Credentials. Access Denied.")
else:
    # --- INVESTIGATION DASHBOARD ---
    st.sidebar.markdown(f"**⚡ OPERATIVE: SANSKAR**")
    st.sidebar.markdown("**📍 LOCATION: NAGPUR_MS_IN**")
    st.sidebar.markdown("---")
    
    @st.cache_resource
    def load_forensic_engine():
        model_path = 'forgery_detector.h5'
        if os.path.exists(model_path):
            return load_model(model_path, compile=False)
        return None
    
    model = load_forensic_engine()

    if model is None:
        st.error("❌ Model file 'forgery_detector.h5' not found in root directory.")

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
                # Temporary file handling
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b:
                    b.write(f.getbuffer())
                
                # 1. Pre-process (Strict 128x128)
                img_data = prepare_image_for_cnn(t_path)
                
                # 2. Add Batch Dimension (Fix for ValueError)
                tensor = np.expand_dims(img_data, axis=0)
                
                # 3. Predict
                pred = model.predict(tensor, verbose=0)[0][0]
                
                os.remove(t_path)
                results.append({
                    "FILENAME": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%"
                })
                bar.progress((idx + 1) / len(files))
            
            st.markdown("### 📊 FINAL DETERMINATION REPORT")
            st.table(pd.DataFrame(results))

    if st.sidebar.button("🔴 EXIT SESSION"):
        st.session_state["logged_in"] = False
        st.rerun()