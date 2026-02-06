import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- PAGE CONFIG ---
st.set_page_config(page_title="ForensiX | Security Portal", layout="wide", page_icon="🕵️")

# --- INITIALIZE SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- CYBER-LOGIN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', Courier, monospace; }
    
    /* Login Box */
    .login-container {
        max-width: 400px;
        margin: auto;
        padding: 40px;
        background: #0f1116;
        border: 2px solid #00f2ff;
        box-shadow: 0px 0px 30px rgba(0, 242, 255, 0.2);
        border-radius: 10px;
        text-align: center;
    }
    
    /* Input Styling */
    input {
        background-color: #050607 !important;
        color: #00f2ff !important;
        border: 1px solid #00f2ff !important;
    }
    
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00f2ff;
        color: #000;
        box-shadow: 0px 0px 20px #00f2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN SCREEN ---
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2562/2562102.png", width=100)
        st.markdown("## SECURITY CLEARANCE")
        
        username = st.text_input("AGENT ID")
        password = st.text_input("ACCESS KEY", type="password")
        
        if st.button("AUTHORIZE"):
            # You can change these credentials
            if username == "Sanskar" and password == "detective2026":
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("ACCESS DENIED: INVALID CREDENTIALS")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APPLICATION (ONLY RUNS IF LOGGED IN) ---
def main_app():
    # --- MODEL LOADING ---
    @st.cache_resource
    def get_model():
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'forgery_detector.h5')
        return load_model(model_path) if os.path.exists(model_path) else None

    model = get_model()

    # --- SIDEBAR & LOGOUT ---
    with st.sidebar:
        st.markdown(f"### ⚡ AGENT ACTIVE")
        st.info("Status: Authorized")
        if st.button("LOGOUT"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="EXBT-ALPHA-01")
        case_notes = st.text_area("CASE CONCLUSION", height=200)

    # ... [Insert your existing Main Dashboard code here] ...
    st.markdown("<h1>🛰️ FORENSIX: TACTICAL SCANNER</h1>")
    
    uploaded_files = st.file_uploader("UPLOAD DIGITAL EVIDENCE", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("RUN DEEP_SCAN INTERROGATION"):
            # [The interrogation logic you already have]
            st.success("Interrogation Complete")

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    login_page()
else:
    main_app()