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
from metadata_scanner import scan_metadata # Utilizes exiftool or PIL
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

# --- CORE UTILITIES ---
def get_timestamp():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def log_forensic_action(action):
    entry = f"[{get_timestamp()}] {action}"
    st.session_state["case_log"].append(entry)

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def generate_heatmap(original_img_bytes, ela_img):
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width, _ = original.shape
    ela_cv = np.array(ela_img.convert('RGB'))
    gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)
    heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_color, (width, height))
    return cv2.addWeighted(original, 0.6, heatmap_resized, 0.4, 0)

# --- FORGERY TYPE CLASSIFIER ---
def classify_forgery_type(ela_img):
    """Analyzes ELA pixel intensity to suspect forgery technique."""
    ela_array = np.array(ela_img.convert('L'))
    # [cite_start]Thresholding high-intensity error regions (outliers) [cite: 231]
    high_error_pixels = np.count_nonzero(ela_array > 40)
    total_pixels = ela_array.size
    density = high_error_pixels / total_pixels

    if density > 0.15:
        return "🚩 Splicing (Heavy Manipulation)"
    elif 0.05 < density <= 0.15:
        return "🚩 Copy-Move (Localized Artifacts)"
    else:
        return "🏳️ Standard Compression"

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

# --- UI CSS ---
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
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
        }
        [data-testid="stForm"] { border: none !important; padding: 0 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
        .dossier-header { background-color: #00f2ff; color: #000; padding: 5px 15px; font-weight: bold; font-size: 11px; border-radius: 5px 5px 0 0; letter-spacing: 1.5px; display: inline-block; }
        .dossier-box { background: rgba(25, 27, 32, 0.95) !important; border: 1px solid #00f2ff !important; border-radius: 0 5px 5px 5px; padding: 10px; }
        </style>
        """, unsafe_allow_html=True)

# --- APP LOGIC ---
if not st.session_state["logged_in"]:
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
            if c1.button("New Registration", use_container_width=True):
                st.session_state["auth_mode"] = "register"; st.rerun()
            if c2.button("Forgot Password", use_container_width=True):
                st.session_state["auth_mode"] = "forgot"; st.rerun()
        elif st.session_state["auth_mode"] == "register":
            with st.form("reg_form"):
                st.markdown("### Agent Enrollment")
                new_u = st.text_input("SET AGENT ID")
                new_p = st.text_input("SET ACCESS KEY", type="password")
                new_r = st.text_input("RECOVERY HINT")
                if st.form_submit_button("ENROLL AGENT", use_container_width=True):
                    if new_u and new_p:
                        hp, hr = hashlib.sha256(new_p.encode()).hexdigest(), hashlib.sha256(new_r.encode()).hexdigest()
                        conn = sqlite3.connect('users.db'); c = conn.cursor()
                        try:
                            c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u.lower().strip(), hp, hr))
                            conn.commit(); st.success("Success! Please Login."); st.session_state["auth_mode"] = "login"; st.rerun()
                        except: st.error("ID exists."); conn.close()
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()
        elif st.session_state["auth_mode"] == "forgot":
            with st.form("recovery_form"):
                st.markdown("### Credential Recovery")
                f_u = st.text_input("AGENT ID")
                f_r = st.text_input("RECOVERY HINT")
                f_np = st.text_input("NEW ACCESS KEY", type="password")
                if st.form_submit_button("RESET ACCESS KEY", use_container_width=True):
                    hr = hashlib.sha256(f_r.encode()).hexdigest()
                    conn = sqlite3.connect('users.db'); c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE username=? AND recovery=?", (f_u.lower().strip(), hr))
                    if c.fetchone():
                        hp = hashlib.sha256(f_np.encode()).hexdigest()
                        c.execute("UPDATE users SET password=? WHERE username=?", (hp, f_u.lower().strip()))
                        conn.commit(); st.success("Reset Successful!"); st.session_state["auth_mode"] = "login"; st.rerun()
                    else: st.error("Recovery hint mismatch.")
                    conn.close()
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Image Forgery Detector</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        if os.path.exists(mp) and os.path.getsize(mp) > 2000:
            return load_model(mp, compile=False) 
        return None
    model = get_model()

    with st.sidebar:
        st.markdown(f"""<div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 25px;">
            <h4 style="margin:0; font-size: 14px; opacity: 0.8;">OPERATIVE STATUS</h4>
            <h2 style="margin:0; color: #00f2ff; font-size: 22px;">⚡ {st.session_state['user'].upper()}</h2>
            <p style="margin:10px 0 0 0; font-size: 14px; color: #00f2ff; font-weight: bold;">📍 LOCATION: NAGPUR_MS_IN</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 📜 SESSION LOG")
        with st.expander("Chain of Custody", expanded=False):
            for entry in st.session_state["case_log"]: st.text(entry)
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        st.markdown('<div class="dossier-header">📝 INVESTIGATION LOG</div><div class="dossier-box">', unsafe_allow_html=True)
        case_notes = st.text_area("FIELD NOTES", height=150, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        for f in files:
            f_hash = get_file_hash(f.getvalue())
            log_forensic_action(f"Exhibit {f.name} logged.")
            st.info(f"🧬 EXHIBIT {f.name} | HASH: {f_hash}")
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f, quality=90)
            heat_img = generate_heatmap(f.getvalue(), ela_img)
            with c_o: st.image(f, caption="SOURCE EVIDENCE")
            with c_h: st.image(heat_img, caption="HEATMAP ANALYSIS")

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 SCANNING PIXELS...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        # 1. ENHANCED METADATA SCAN
                        has_meta, meta_info = scan_metadata(tmp)
                        software_tag = "🏳️ ORIGINAL"
                        software_list = ["Adobe Photoshop", "Canva", "GIMP", "PicsArt", "Lightroom"]
                        if has_meta and any(s.lower() in str(meta_info).lower() for s in software_list):
                            software_tag = "🚩 EDITED (Software Tags Found)"
                        
                        # 2. SUSPECTED FORGERY TYPE (BY ELA DENSITY)
                        ela_img_data = convert_to_ela_image(f, quality=90)
                        forgery_type = classify_forgery_type(ela_img_data)
                        
                        # [cite_start]3. CNN PREDICTION [cite: 131, 132]
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)

                        # Standardized key 'CONFIDENCE' to match report_gen.py requirements
                        results.append({
                            "FILENAME": f.name, 
                            "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                            "FORGERY TYPE": forgery_type,
                            "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                            "METADATA": software_tag
                        })
                        bar.progress((idx+1)/len(files))
                    
                    # Generate PDF with standardized results
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", create_pdf_report(results, case_notes=case_notes))
                    status.update(label="COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.markdown("---")
        st.markdown(f"### 📊 FINAL DETERMINATION REPORT: {case_id}")
        st.table(pd.DataFrame(st.session_state["analysis_results"]))
        st.download_button("📥 DOWNLOAD CASE DOSSIER (.ZIP)", st.session_state["zip_buffer"], f"{case_id}.zip", use_container_width=True)

    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""<div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
            <span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
            <span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span>
        </div>""", unsafe_allow_html=True)
    sync_clock()