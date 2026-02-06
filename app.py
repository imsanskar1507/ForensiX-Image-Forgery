import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

st.set_page_config(page_title="ForensiX | Detective Suite", layout="wide", page_icon="🕵️")

# Custom UI
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle, #1a1c23 0%, #0d0e12 100%); color: #e0e0e0; font-family: 'Courier New', Courier, monospace; }
    .evidence-card { background: rgba(25, 28, 35, 0.8); border: 1px solid rgba(139, 0, 0, 0.3); border-left: 5px solid #8b0000; border-radius: 8px; padding: 20px; margin-bottom: 20px; backdrop-filter: blur(10px); }
    h1 { color: #f0f6fc; text-shadow: 0px 0px 10px rgba(139, 0, 0, 0.7); letter-spacing: 4px; border-bottom: 1px solid #8b0000; }
    .status-text { color: #8b0000; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    .stButton>button { width: 100%; background: linear-gradient(145deg, #8b0000, #5a0000); color: white; border-radius: 4px; height: 3.5rem; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'forgery_detector.h5')
    return load_model(model_path) if os.path.exists(model_path) else None

model = get_model()

with st.sidebar:
    st.markdown("### 🕵️ CASE DOSSIER")
    case_ref = st.text_input("CASE ID", value="REF-9921-X")
    investigator = st.text_input("AGENT", value="Sanskar Dhore")
    case_notes = st.text_area("INVESTIGATOR NOTES", placeholder="Enter manual summary...", height=150)

st.markdown("<h1>📁 FORENSIC INTERROGATION SUITE</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='status-text'>SYSTEM STATUS: ACTIVE // AGENT: {investigator.upper()}</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📥 SUBMIT EVIDENCE", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("⚡ BEGIN FORENSIC SCAN"):
        results = []
        progress_bar = st.progress(0)
        with st.status("🕵️ ANALYZING SCENE...", expanded=True) as status:
            for idx, file in enumerate(uploaded_files):
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f: f.write(file.getbuffer())
                
                meta_data, meta_msg = scan_metadata(temp_path)
                processed = prepare_image_for_cnn(temp_path)
                pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
                os.remove(temp_path)
                
                verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                results.append({"ID": f"EXBT-{idx+1001}", "FILENAME": file.name, "VERDICT": verdict, "CONFIDENCE": float(max(pred, 1-pred)*100), "METADATA": meta_msg})
                progress_bar.progress((idx + 1) / len(uploaded_files))
            status.update(label="SCAN COMPLETE", state="complete")

        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=300, hide_index=True, column_config={"CONFIDENCE": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100)})
        
        pdf_bytes = create_pdf_report(results, case_notes=case_notes)
        st.download_button("📥 DOWNLOAD OFFICIAL DOSSIER", pdf_bytes, f"{case_ref}_Report.pdf", "application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)