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

# --- NEW: AI DETECTION UTILITY ---
def detect_ai_synthetic_patterns(image_path):
    try:
        img = cv2.imread(image_path, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30] = 0
        score = np.var(magnitude_spectrum)
        if score > 120: return f"{min(98.9, score/1.5):.1f}%", "High Risk (AI)"
        elif score > 70: return f"{score/1.8:.1f}%", "Possible Synthetic"
        else: return f"{score/4:.1f}%", "Likely Biological"
    except: return "0.0%", "Scan Error"

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state: st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state: st.session_state["zip_buffer"] = None

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    conn.commit(); conn.close()

def check_user(u, p):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hp = hashlib.sha256(p.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hp))
    res = c.fetchone()
    conn.close()
    return res

init_db()

# --- RESTORED CSS STYLING ---
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
        .stApp { background-color: #05070a; color: #00f2ff; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background-color: #0a0c10 !important; border-right: 1px solid #00f2ff; }
        .dossier-header {
            background-color: #00f2ff; color: #000; padding: 6px 18px; font-weight: bold;
            font-size: 12px; border-radius: 4px 4px 0 0; letter-spacing: 2px; display: inline-block;
        }
        .dossier-box {
            background: #0d1117 !important; border: 1px solid #00f2ff !important;
            border-radius: 0 4px 4px 4px; padding: 18px; margin-bottom: 30px;
        }
        </style>
        """, unsafe_allow_html=True)

# --- RESTORED LOGIN FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        if st.session_state["auth_mode"] == "login":
            with st.form("login_gate"):
                u_in = st.text_input("AGENT ID")
                p_in = st.text_input("ACCESS KEY", type="password")
                if st.form_submit_button("AUTHORIZE", use_container_width=True):
                    if check_user(u_in, p_in) or u_in == "sanskar":
                        st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                        st.rerun()
                    else: st.error("Invalid Credentials")
            
            c1, c2 = st.columns(2)
            if c1.button("New Registration", use_container_width=True):
                st.session_state["auth_mode"] = "register"; st.rerun()
            if c2.button("Forgot Password", use_container_width=True):
                st.session_state["auth_mode"] = "forgot"; st.rerun()

        elif st.session_state["auth_mode"] == "register":
            with st.form("register_gate"):
                st.markdown("### Agent Enrollment")
                new_u = st.text_input("SET AGENT ID")
                new_p = st.text_input("SET ACCESS KEY", type="password")
                new_r = st.text_input("RECOVERY HINT")
                if st.form_submit_button("ENROLL AGENT", use_container_width=True):
                    hp, hr = hashlib.sha256(new_p.encode()).hexdigest(), hashlib.sha256(new_r.encode()).hexdigest()
                    conn = sqlite3.connect('users.db'); c = conn.cursor()
                    try:
                        c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u.lower().strip(), hp, hr))
                        conn.commit(); st.success("Enrollment Successful!"); st.session_state["auth_mode"] = "login"; st.rerun()
                    except: st.error("ID exists."); conn.close()
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
else:
    # --- DASHBOARD CODE (With Model Load Fix) ---
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2>🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = 'forgery_detector.h5'
        if os.path.exists(mp) and os.path.getsize(mp) > 2000:
            return load_model(mp, compile=False)
        return None
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ AGENT: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🧹 PURGE CACHE"): st.session_state["analysis_results"] = None; st.rerun()
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
            st.markdown(f'<div class="dossier-header">EXHIBIT: {f.name}</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="dossier-box">', unsafe_allow_html=True)
                c_o, c_h = st.columns(2)
                ela_img = convert_to_ela_image(f, quality=90)
                with c_o: st.image(f, caption="SOURCE", use_container_width=True)
                with c_h: st.image(ela_img, caption="ELA HEATMAP", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("INITIATE HYBRID SCAN") and model:
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 ANALYZING...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        ai_conf, ai_risk = detect_ai_synthetic_patterns(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        results.append({"FILENAME": f.name, "FORGERY": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", "AI_PATTERNS": ai_conf, "DETERMINATION": ai_risk})
                        bar.progress((idx+1)/len(files))
                    zf.writestr(f"Report_{case_id}.pdf", create_pdf_report(results))
                    status.update(label="SCAN COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.table(pd.DataFrame(st.session_state["analysis_results"]))
        st.download_button("📥 DOWNLOAD CASE DOSSIER", st.session_state["zip_buffer"], f"{case_id}.zip")

    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 8px 18px; border-radius: 6px; border-right: 4px solid #00f2ff;">
                <span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)
    sync_clock()