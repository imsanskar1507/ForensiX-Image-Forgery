import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from keras.models import load_model
from report_gen import create_pdf_report 

# 1. Page Config
st.set_page_config(page_title="ForensiX: Case Manager", layout="wide", page_icon="🕵️")

# 2. Detective Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0d0e12; color: #d1d1d1; font-family: 'Courier New', Courier, monospace; }
    .evidence-card {
        background: #161a21; border: 1px solid #2d333b; border-left: 10px solid #8b0000;
        border-radius: 4px; padding: 25px; margin-bottom: 25px; box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
    }
    h1 { color: #f0f6fc; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid #8b0000; padding-bottom: 10px; }
    h3 { color: #8b0000; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 3px; }
    .stButton>button {
        width: 100%; background-color: #8b0000; color: white; border: 1px solid #ff0000;
        border-radius: 2px; height: 3.5rem; font-weight: 700; text-transform: uppercase; letter-spacing: 5px; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff0000; box-shadow: 0px 0px 20px rgba(255, 0, 0, 0.4); color: black; }
    [data-testid="stSidebar"] { background-color: #0a0c10; border-right: 2px solid #2d333b; padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model('forgery_detector.h5')

model = get_model()

# --- SIDEBAR: CASE MANAGEMENT ---
with st.sidebar:
    st.markdown("### 🕵️ CASE DOSSIER")
    case_name = st.text_input("CASE REFERENCE", value="EXBT-2026-ALPHA")
    investigator = st.text_input("LEAD INVESTIGATOR", value="Sanskar Dhore")
    case_priority = st.select_slider("PRIORITY", options=["LOW", "MED", "HIGH", "URGENT"])
    
    st.markdown("---")
    st.markdown("### 📝 FINAL CONCLUSION")
    # This stores your manual notes
    case_notes = st.text_area("Investigator's Summary", 
                              placeholder="Describe findings, suspect patterns, or case wrap-up...",
                              height=150)
    
    st.markdown("---")
    if st.button("RESET INVESTIGATION"):
        st.rerun()

# --- CASE HEADER ---
st.markdown(f"<h1>📁 CASE: {case_name}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #8b0000; font-weight: bold;'>INVESTIGATOR: {investigator} // PRIORITY: {case_priority}</p>", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
uploaded_files = st.file_uploader("DRAG & DROP EXHIBITS", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            st.markdown("### EXHIBIT A: SOURCE")
            st.image(Image.open(uploaded_files[0]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            st.markdown("### EXHIBIT B: ELA MAP")
            ela_img = convert_to_ela_image(uploaded_files[0], quality=90)
            st.image(ela_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("INITIATE FORENSIC INTERROGATION"):
        results = []
        progress_bar = st.progress(0)
        
        with st.expander("🕵️ SYSTEM LOG: LIVE FEED", expanded=True):
            log_placeholder = st.empty()
            log_entries = []

        for idx, file in enumerate(uploaded_files):
            log_entries.append(f"🔍 Scanning Exhibit {idx+1}: {file.name}...")
            log_placeholder.code("\n".join(log_entries[-5:]))
            
            temp_path = f"temp_{file.name}"
            Image.open(file).save(temp_path)
            
            meta_data, meta_msg = scan_metadata(temp_path)
            processed = prepare_image_for_cnn(temp_path)
            pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
            os.remove(temp_path)
            
            verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
            results.append({
                "CASE_ID": f"EXBT-{idx+1001}",
                "FILENAME": file.name,
                "VERDICT": verdict,
                "PROBABILITY": float(max(pred, 1-pred)*100), 
                "METADATA_TRACE": meta_msg
            })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        # --- RESULTS LOG ---
        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
        st.markdown("### 📊 CENTRAL EVIDENCE DATABASE")
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=400, hide_index=True,
                     column_config={"PROBABILITY": st.column_config.ProgressColumn("Confidence", format="%.2f%%", min_value=0, max_value=100)})

        m1, m2, m3 = st.columns(3)
        m1.metric("TOTAL EXHIBITS", len(results))
        m2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]))
        
        with m3:
            # We pass the case_notes from the sidebar to the report generator
            pdf_bytes = create_pdf_report(results, case_notes=case_notes)
            st.download_button(
                label="📥 DOWNLOAD CASE DOSSIER",
                data=pdf_bytes,
                file_name=f"{case_name}_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)