import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- PAGE CONFIG ---
st.set_page_config(page_title="ForensiX | Digital Detective", layout="wide", page_icon="🕵️")

# --- CUSTOM DETECTIVE NOIR CSS ---
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background: radial-gradient(circle, #1a1c23 0%, #0d0e12 100%);
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Glassmorphism Evidence Cards */
    .evidence-card {
        background: rgba(25, 28, 35, 0.8);
        border: 1px solid rgba(139, 0, 0, 0.3);
        border-left: 5px solid #8b0000;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* Neon Red Headers */
    h1 {
        color: #f0f6fc;
        text-shadow: 0px 0px 10px rgba(139, 0, 0, 0.7);
        letter-spacing: 4px;
        text-transform: uppercase;
        border-bottom: 1px solid #8b0000;
    }
    
    .status-text {
        color: #8b0000;
        font-weight: bold;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }

    /* Tactical Sidebar */
    [data-testid="stSidebar"] {
        background-color: #08090b;
        border-right: 1px solid #2d333b;
    }

    /* Big Red Tactical Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(145deg, #8b0000, #5a0000);
        color: white;
        border: 1px solid #ff4b4b;
        border-radius: 4px;
        height: 4rem;
        font-weight: 900;
        letter-spacing: 3px;
        transition: 0.4s all;
    }
    .stButton>button:hover {
        background: #ff0000;
        box-shadow: 0px 0px 25px rgba(255, 0, 0, 0.5);
        transform: scale(1.02);
    }

    /* Styled Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #8b0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def get_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'forgery_detector.h5')
    if not os.path.exists(model_path):
        return None
    try:
        return load_model(model_path)
    except:
        return None

model = get_model()

# --- SIDEBAR CASE MANAGEMENT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2562/2562102.png", width=80)
    st.markdown("### 🕵️ DOSSIER CONTROL")
    case_ref = st.text_input("CASE ID", value="REF-9921-X")
    investigator = st.text_input("AGENT NAME", value="Sanskar Dhore")
    
    st.markdown("---")
    st.markdown("### 📝 AGENT NOTES")
    case_notes = st.text_area("Observations", placeholder="Enter field notes here...", height=150)
    
    if st.button("🚨 WIPE EVIDENCE"):
        st.rerun()

# --- MAIN UI HEADER ---
st.markdown("<h1>📁 FORENSIC INTERROGATION SUITE</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='status-text'>SYSTEM STATUS: ACTIVE // ENCRYPTED CONNECTION // AGENT: {investigator.upper()}</p>", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("📥 DROP EVIDENCE SAMPLES HERE", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    if len(uploaded_files) == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="evidence-card"><h3>EXHIBIT: SOURCE</h3>', unsafe_allow_html=True)
            st.image(uploaded_files[0], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="evidence-card"><h3>EXHIBIT: ELA MAP</h3>', unsafe_allow_html=True)
            ela_img = convert_to_ela_image(uploaded_files[0], quality=90)
            st.image(ela_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- ACTION BUTTON ---
    if st.button("⚡ BEGIN SCANNING PROCESS"):
        if model is None:
            st.error("SYSTEM ERROR: Forensic Engine (Model) not found.")
        else:
            results = []
            progress_bar = st.progress(0)
            
            with st.status("🕵️ CROSS-REFERENCING DATA...", expanded=True) as status:
                log_placeholder = st.empty()
                for idx, file in enumerate(uploaded_files):
                    log_placeholder.text(f"Scanning Exhibit {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Processing
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
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
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                status.update(label="SCAN COMPLETE", state="complete", expanded=False)

            # --- RESULTS LOG ---
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            st.markdown("### 📊 INVESTIGATION LOG")
            df = pd.DataFrame(results)
            st.dataframe(
                df, 
                use_container_width=True, 
                height=350,
                column_config={
                    "CONFIDENCE": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100),
                    "VERDICT": st.column_config.TextColumn("Status")
                },
                hide_index=True
            )

            # Summary Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("TOTAL SCANNED", len(results))
            m2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]), delta_color="inverse")
            
            with m3:
                pdf_bytes = create_pdf_report(results, case_notes=case_notes)
                st.download_button(
                    label="📥 DOWNLOAD DOSSIER",
                    data=pdf_bytes,
                    file_name=f"{case_ref}_Report.pdf",
                    mime="application/pdf"
                )
            st.markdown('</div>', unsafe_allow_html=True)