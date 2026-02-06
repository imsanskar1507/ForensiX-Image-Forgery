import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Security Portal", layout="wide", page_icon="🕵️")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- TACTICAL CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', Courier, monospace; }
    .login-box {
        max-width: 450px; margin: auto; padding: 50px;
        background: #0f1116; border: 2px solid #00f2ff;
        box-shadow: 0px 0px 30px rgba(0, 242, 255, 0.3); border-radius: 10px;
        text-align: center;
    }
    .evidence-card {
        background: #0f1116; border: 2px solid #00f2ff;
        box-shadow: 0px 0px 15px #00f2ff; border-radius: 10px;
        padding: 25px; margin-bottom: 25px;
    }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 0px 0px 10px #00f2ff; text-transform: uppercase; }
    .stButton>button {
        width: 100%; background-color: transparent; color: #00f2ff;
        border: 2px solid #00f2ff; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #00f2ff; color: #000; box-shadow: 0px 0px 20px #00f2ff; }
    input { background-color: #050607 !important; color: #00f2ff !important; border: 1px solid #00f2ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN VIEW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h1>🔒 CORE ACCESS</h1>", unsafe_allow_html=True)
        user = st.text_input("AGENT ID", placeholder="Enter ID...")
        pw = st.text_input("ACCESS KEY", type="password", placeholder="Enter Key...")
        
        if st.button("AUTHORIZE SYSTEM"):
            if user.strip().lower() == "sanskar" and pw.strip() == "detective2026":
                st.session_state["logged_in"] = True
                with st.spinner("DECRYPTING INTERFACE..."):
                    time.sleep(1.5)
                st.rerun()
            else:
                st.error("ACCESS DENIED: CREDENTIALS NOT RECOGNIZED")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APP VIEW ---
else:
    @st.cache_resource
    def get_model():
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'forgery_detector.h5')
        return load_model(model_path) if os.path.exists(model_path) else None

    model = get_model()

    with st.sidebar:
        st.markdown("### 🖥️ AGENT ACTIVE")
        st.success(f"User: {st.session_state.get('user', 'Sanskar')}")
        if st.button("TERMINATE SESSION"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="EXBT-ALPHA-01")
        case_notes = st.text_area("INVESTIGATOR NOTES", height=200)

    st.markdown("<h1>🛰️ FORENSIX: TACTICAL SCANNER</h1>", unsafe_allow_html=True)
    
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        if st.button("RUN DEEP_SCAN INTERROGATION"):
            results = []
            progress = st.progress(0)
            
            with st.status("📡 ANALYZING SCENE...", expanded=True) as status:
                for idx, file in enumerate(files):
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f: f.write(file.getbuffer())
                    
                    meta_data, meta_msg = scan_metadata(temp_path)
                    processed = prepare_image_for_cnn(temp_path)
                    pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
                    os.remove(temp_path)
                    
                    verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                    results.append({
                        "FILENAME": file.name, "VERDICT": verdict, 
                        "CONFIDENCE": float(max(pred, 1-pred)*100), "METADATA": meta_msg
                    })
                    progress.progress((idx + 1) / len(files))
                status.update(label="SCAN COMPLETE", state="complete")

            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("SCANNED", len(results))
            c2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]))
            with c3:
                pdf_bytes = create_pdf_report(results, case_notes=case_notes)
                st.download_button("📥 EXPORT DOSSIER", pdf_bytes, f"REPORT_{case_id}.pdf")
            st.markdown('</div>', unsafe_allow_html=True)