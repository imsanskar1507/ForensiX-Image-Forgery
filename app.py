import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pytz 
import pandas as pd
import sqlite3
import hashlib
import cv2
import io
import zipfile
import time 
import matplotlib.pyplot as plt
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state:
    st.session_state["case_log"] = []
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state:
    st.session_state["zip_buffer"] = None

# --- DATABASE & UTILS (Preserved from original) ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit(); conn.close()

init_db()

# --- CSS STYLING ---
if not st.session_state["logged_in"]:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(10, 11, 13, 0.85), rgba(10, 11, 13, 0.95)), 
                        url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
            background-size: cover; background-attachment: fixed; color: #00f2ff;
        }
        .login-box {
            background: rgba(15, 17, 22, 0.75) !important; backdrop-filter: blur(15px);
            border: 2px solid #00f2ff; border-radius: 15px; padding: 25px;
        }
        [data-testid="stForm"] { border: none !important; padding: 0 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
        .report-frame { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; border-radius: 10px; padding: 20px; margin-top: 25px; }
        </style>
        """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    # (Registration/Forgot/Login Logic here using st.form for Enter Key support)
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Login</h1>", unsafe_allow_html=True)
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                hp = hashlib.sha256(p_in.encode()).hexdigest()
                if u_in == "sanskar" and hp == hashlib.sha256("detective2026".encode()).hexdigest():
                    st.session_state["logged_in"] = True; st.session_state["user"] = u_in; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD ---
    col_t, col_c = st.columns([2, 1])
    with col_t: st.markdown('<h2>🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_c: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = os.path.join(os.path.dirname(__file__), 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ AGENT: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("FIELD NOTES", height=150)
        if st.button("🧹 CLEAR RESULTS"):
            st.session_state["analysis_results"] = None; st.session_state["zip_buffer"] = None; st.rerun()
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        if st.button("INITIATE DEEP SCAN"):
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 ANALYZING...", expanded=True) as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        has_meta, m_msg = scan_metadata(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        v = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                        # Pass string-formatted confidence to result list
                        results.append({
                            "FILENAME": f.name, 
                            "VERDICT": v, 
                            "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                            "METADATA": "DETECTED" if has_meta else "NONE"
                        })
                        bar.progress((idx+1)/len(files))
                    
                    pdf_data = create_pdf_report(results, case_notes=case_notes)
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", pdf_data)
                    status.update(label="SCAN COMPLETE", state="complete")
            
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    # --- PERSISTENT REPORT SECTION ---
    if st.session_state["analysis_results"]:
        st.markdown("---")
        st.markdown("### 📊 FINAL DETERMINATION REPORT")
        with st.container():
            st.markdown('<div class="report-frame">', unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state["analysis_results"]))
            st.download_button(
                label="📥 DOWNLOAD CASE DOSSIER (.ZIP)",
                data=st.session_state["zip_buffer"],
                file_name=f"CASE_{case_id}_ANALYSIS.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # --- CLOCK FRAGMENT ---
    while st.session_state["logged_in"]:
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
                <span style="color: #00f2ff; font-size: 16px; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)