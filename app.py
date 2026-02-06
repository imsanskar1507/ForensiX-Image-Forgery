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
st.set_page_config(page_title="ForensiX | Tactical Suite", layout="wide", page_icon="🕵️")

# --- CYBER-FORENSICS CSS ---
st.markdown("""
    <style>
    /* Dark Terminal Background */
    .stApp {
        background-color: #0a0b0d;
        color: #00f2ff;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Glowing Neon Card */
    .evidence-card {
        background: #0f1116;
        border: 2px solid #00f2ff;
        box-shadow: 0px 0px 15px #00f2ff;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
    }

    /* Neon Headers */
    h1, h2, h3 {
        color: #00f2ff !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0px 0px 10px #00f2ff;
    }

    /* Red Alert Text for Forgery */
    .status-alert {
        color: #ff003c;
        font-weight: bold;
        text-shadow: 0px 0px 5px #ff003c;
    }

    /* Tactical Sidebar */
    [data-testid="stSidebar"] {
        background-color: #050607;
        border-right: 1px solid #00f2ff;
    }

    /* Cyan Cyber Button */
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        border-radius: 0px;
        height: 3.5rem;
        font-weight: bold;
        text-transform: uppercase;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00f2ff;
        color: #000;
        box-shadow: 0px 0px 20px #00f2ff;
    }

    /* Data Table Styling */
    .stDataFrame {
        border: 1px solid #00f2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'forgery_detector.h5')
    return load_model(model_path) if os.path.exists(model_path) else None

model = get_model()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🖥️ SYSTEM CONFIG")
    case_id = st.text_input("CASE ID", value="EXBT-ALPHA-01")
    agent = st.text_input("INVESTIGATOR", value="Sanskar Dhore")
    st.markdown("---")
    case_notes = st.text_area("CASE CONCLUSION", placeholder="Enter final summary...", height=200)

# --- MAIN DASHBOARD ---
st.markdown("<h1>🛰️ ForeniX: Image Forgery</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#00f2ff;'>ACTIVE SESSION // AGENT_{agent.upper().replace(' ', '_')}</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("UPLOAD DIGITAL EVIDENCE", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) == 1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="evidence-card"><h3>SOURCE_FILE</h3>', unsafe_allow_html=True)
            st.image(uploaded_files[0], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="evidence-card"><h3>ELA_ANALYSIS</h3>', unsafe_allow_html=True)
            ela = convert_to_ela_image(uploaded_files[0], quality=90)
            st.image(ela, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("RUN DEEP_SCAN INTERROGATION"):
        results = []
        progress = st.progress(0)
        
        with st.status("📡 INTERROGATING DATASTREAMS...", expanded=True) as status:
            for idx, file in enumerate(uploaded_files):
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f: f.write(file.getbuffer())
                
                meta_data, meta_msg = scan_metadata(temp_path)
                processed = prepare_image_for_cnn(temp_path)
                pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
                os.remove(temp_path)
                
                verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                results.append({
                    "ID": f"EXBT-{idx+1001}", 
                    "FILENAME": file.name, 
                    "VERDICT": verdict, 
                    "CONFIDENCE": float(max(pred, 1-pred)*100), 
                    "METADATA": meta_msg
                })
                progress.progress((idx + 1) / len(uploaded_files))
            status.update(label="SCAN COMPLETE", state="complete")

        # --- LOG DISPLAY ---
        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
        st.markdown("### 📋 INVESTIGATION DATABASE")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=350, hide_index=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("TOTAL SCANNED", len(results))
        m2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]))
        
        with m3:
            pdf_bytes = create_pdf_report(results, case_notes=case_notes)
            st.download_button("📥 EXPORT DOSSIER", pdf_bytes, f"REPORT_{case_id}.pdf", "application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)