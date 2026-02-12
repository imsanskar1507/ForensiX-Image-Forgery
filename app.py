import streamlit as st
import numpy as np
import os
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide")

# Force initialization of session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN INTERFACE ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    
    with col:
        st.info("Agent Authentication Required")
        user_id = st.text_input("AGENT ID")
        access_key = st.text_input("ACCESS KEY", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("AUTHORIZE SESSION", use_container_width=True):
                # Hardcoded check for Sanskar Dhore's profile
                if user_id == "sanskar" and access_key == "detective2026":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        
        with col2:
            # Emergency Bypass Button (for testing)
            if st.button("QUICK LOGIN (DEBUG)", use_container_width=True):
                st.session_state.logged_in = True
                st.rerun()

        st.markdown("---")
        st.caption("Nagpur Division - Forensic Image Analysis Tool")

else:
    # --- INVESTIGATION DASHBOARD ---
    st.sidebar.markdown(f"**⚡ OPERATIVE: SANSKAR**")
    st.sidebar.markdown("**📍 LOCATION: NAGPUR_MS_IN**")
    
    if st.sidebar.button("🔴 EXIT SESSION"):
        st.session_state.logged_in = False
        st.rerun()

    @st.cache_resource
    def load_engine():
        path = 'forgery_detector.h5'
        if os.path.exists(path):
            return load_model(path, compile=False)
        return None
    
    model = load_engine()

    st.markdown("## 🛰️ Forensic Evidence Analysis")
    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        for f in files:
            col1, col2 = st.columns(2)
            with col1: st.image(f, caption="SOURCE", use_container_width=True)
            with col2: st.image(convert_to_ela_image(f), caption="ELA ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            for f in files:
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b: b.write(f.getbuffer())
                
                # Pre-processing (128x128 Fix)
                img_data = prepare_image_for_cnn(t_path)
                tensor = np.expand_dims(img_data, axis=0)
                
                # Predict
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_path)
                
                results.append({
                    "FILENAME": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%"
                })
            
            st.table(pd.DataFrame(results))