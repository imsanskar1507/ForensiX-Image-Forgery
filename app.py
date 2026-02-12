import streamlit as st
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pytz
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Force persistent session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- UI BRANDING ---
st.markdown("""
<style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
    .stButton>button { background-color: transparent; color: #00f2ff; border: 1px solid #00f2ff; width: 100%; border-radius: 0px; }
    .stButton>button:hover { background-color: #00f2ff; color: #000; }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION INTERFACE ---
if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.write("Enter Credentials for Nagpur Unit Terminal")
        u_in = st.text_input("AGENT ID")
        p_in = st.text_input("ACCESS KEY", type="password")
        
        # We use a direct check here to prevent the "Login Problem"
        if st.button("AUTHORIZE SESSION", use_container_width=True):
            if u_in == "sanskar" and p_in == "detective2026":
                st.session_state["logged_in"] = True
                st.success("Authorization Successful. Redirecting...")
                st.rerun()
            else:
                st.error("Access Denied: Invalid Credentials")
else:
    # --- DASHBOARD UI ---
    c1, c2 = st.columns([3, 1])
    with c1: 
        st.markdown('<h2 style="color:#00f2ff; margin:0;">🛰️ Forensic Investigation Dashboard</h2>', unsafe_allow_html=True)
        st.caption("Nagpur Division Forensic Unit | G H Raisoni University")
    with c2:
        now = datetime.now(IST).strftime('%I:%M:%S %p')
        st.markdown(f"<p style='text-align:right; font-size:18px;'>🕒 {now}</p>", unsafe_allow_html=True)

    # Sidebar operative status
    with st.sidebar:
        st.markdown(f"""<div style="border: 1px solid #00f2ff; padding: 15px; border-radius: 5px; background: rgba(0, 242, 255, 0.05);">
            <p style="margin:0; font-size: 10px; opacity: 0.6;">OPERATIVE STATUS</p>
            <h3 style="margin:0; color: #00f2ff;">⚡ SANSKAR</h3>
            <p style="margin:0; font-size: 12px;">📍 NAGPUR_MS_IN</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🔴 EXIT SESSION"):
            st.session_state["logged_in"] = False
            st.rerun()

    # Model Loading
    @st.cache_resource
    def load_engine():
        return load_model('forgery_detector.h5', compile=False) if os.path.exists('forgery_detector.h5') else None
    
    model = load_engine()

    # Evidence Upload & Display
    files = st.file_uploader("UPLOAD EXHIBIT EVIDENCE", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if files:
        for f in files:
            st.markdown(f"**FILE: `{f.name}`**")
            
            # Generate and Rewind
            ela_heatmap = convert_to_ela_image(f)
            
            col_a, col_b = st.columns(2)
            with col_a:
                f.seek(0)
                st.image(f, caption="SOURCE EVIDENCE", use_container_width=True)
            with col_b:
                st.image(ela_heatmap, caption="ELA DIFFERENCE MAP", use_container_width=True)

        if st.button("INITIATE DEEP SCAN (CNN)") and model:
            results = []
            for f in files:
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b:
                    f.seek(0)
                    b.write(f.read())
                
                img_data = prepare_image_for_cnn(t_path)
                tensor = np.expand_dims(img_data, axis=0)
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_path)
                
                results.append({
                    "EXHIBIT": f.name, 
                    "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN",
                    "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%"
                })
            st.table(pd.DataFrame(results))