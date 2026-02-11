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
import seaborn as sns
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata 
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session State Initialization
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state: st.session_state["case_log"] = []
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# --- RESEARCH UTILITIES ---
def get_timestamp():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def log_forensic_action(action):
    entry = f"[{get_timestamp()}] {action}"
    st.session_state["case_log"].append(entry)

def classify_forgery_type(ela_img):
    """Analyzes ELA pixel intensity to suspect forgery technique[cite: 231]."""
    ela_array = np.array(ela_img.convert('L'))
    high_error_pixels = np.count_nonzero(ela_array > 40)
    density = high_error_pixels / ela_array.size
    if density > 0.15:
        return "🚩 Splicing (Heavy Manipulation)"
    elif 0.05 < density <= 0.15:
        return "🚩 Copy-Move (Localized Artifacts)"
    else:
        return "🏳️ Standard Compression"

def plot_confusion_matrix():
    """Generates the research-based confusion matrix[cite: 309, 353, 354]."""
    # Performance data from G.H. Raisoni study: 261 TP, 290 TN, 2 FP, 47 FN [cite: 362-365]
    data = [[290, 2], [47, 261]] 
    df_cm = pd.DataFrame(data, index=["REAL", "FAKE"], columns=["REAL", "FAKE"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_title("Research Benchmark Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return fig

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit()
    conn.close()

def check_user(u, p):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hp = hashlib.sha256(p.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hp))
    res = c.fetchone()
    conn.close()
    return res

init_db()

# --- APP LOGIC ---
if not st.session_state["logged_in"]:
    st.markdown("""<style>.stApp { background: linear-gradient(rgba(10, 11, 13, 0.85), rgba(10, 11, 13, 0.95)), url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop"); background-size: cover; color: #00f2ff; } .login-box { background: rgba(15, 17, 22, 0.75); backdrop-filter: blur(15px); border: 2px solid #00f2ff; border-radius: 15px; padding: 25px; }</style>""", unsafe_allow_html=True)
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Image Forgery Detector</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        if st.session_state["auth_mode"] == "login":
            with st.form("login_gate"):
                u_in = st.text_input("AGENT ID")
                p_in = st.text_input("ACCESS KEY", type="password")
                if st.form_submit_button("AUTHORIZE", use_container_width=True):
                    if check_user(u_in, p_in):
                        st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                        log_forensic_action(f"Agent {u_in.upper()} authorized.")
                        st.rerun()
                    else: st.error("Invalid Credentials")
            c1, c2 = st.columns(2)
            if c1.button("Register"): st.session_state["auth_mode"] = "register"; st.rerun()
            if c2.button("Forgot Password"): st.session_state["auth_mode"] = "forgot"; st.rerun()
        # ... Registration/Forgot logic remains unchanged ...
        st.markdown('</div>', unsafe_allow_html=True)
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Dashboard</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        return load_model(mp, compile=False) if os.path.exists(mp) else None
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ OPERATIVE: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 SCANNING PIXELS[cite: 202, 203]...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        has_meta, meta_info = scan_metadata(tmp)
                        software_tag = "🏳️ ORIGINAL"
                        if has_meta and any(s.lower() in str(meta_info).lower() for s in ["Photoshop", "Canva", "GIMP"]):
                            software_tag = "🚩 EDITED"
                        
                        ela_img_data = convert_to_ela_image(f, quality=90)
                        forgery_type = classify_forgery_type(ela_img_data)
                        
                        proc = prepare_image_for_cnn(tmp) # Preprocessing: 224x224 RGB [cite: 129, 193]
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0] # CNN Prediction [cite: 131, 132]
                        os.remove(tmp)

                        results.append({
                            "FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                            "FORGERY TYPE": forgery_type, "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                            "METADATA": software_tag
                        })
                        bar.progress((idx+1)/len(files))
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", create_pdf_report(results, case_notes="Deep Scan complete."))
                    status.update(label="COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.markdown("---")
        st.table(pd.DataFrame(st.session_state["analysis_results"]))
        
        # --- RESEARCH PERFORMANCE VISUALIZATION ---
        st.markdown("### 📈 Research Performance Analysis")
        col_cm, col_metrics = st.columns([1, 1])
        with col_cm:
            st.write("**Model Confusion Matrix [cite: 354, 370]**")
            st.pyplot(plot_confusion_matrix())
        with col_metrics:
            st.write("**G.H. Raisoni Benchmark Results [cite: 23, 376, 456]**")
            st.info(f"""
            - **Accuracy:** 91.4% [cite: 23]
            - **Loss:** 0.342 
            - **AUC Score:** 0.92 [cite: 23, 376]
            - **Dataset Count:** 318 Videos [cite: 272]
            """)
        st.download_button("📥 DOWNLOAD CASE DOSSIER (.ZIP)", st.session_state["zip_buffer"], f"{case_id}.zip", use_container_width=True)

    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""<div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
            <span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
            <span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span>
        </div>""", unsafe_allow_html=True)
    sync_clock()