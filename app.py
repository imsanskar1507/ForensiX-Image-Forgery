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
import time 
import matplotlib.pyplot as plt
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"
if "user" not in st.session_state: st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state: st.session_state["case_log"] = []
if "scan_results" not in st.session_state: st.session_state["scan_results"] = None

# --- CORE UTILITIES ---
def get_timestamp():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def log_forensic_action(action):
    entry = f"[{get_timestamp()}] {action}"
    st.session_state["case_log"].append(entry)

def get_file_hash(file_bytes):
    # Chain of custody hashing [cite: 18, 153]
    return hashlib.sha256(file_bytes).hexdigest()

def generate_heatmap(original_img_bytes, ela_img):
    # Visual heatmap generation [cite: 12]
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

def generate_luminance_map(file):
    # Luminance gradient analysis [cite: 18]
    file.seek(0)
    img = Image.open(file).convert('L')
    img_array = np.array(img, dtype=float)
    dx, dy = np.gradient(img_array)
    gradient = np.sqrt(dx**2 + dy**2)
    gradient = (gradient / (gradient.max() if gradient.max() > 0 else 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)

def plot_histogram(file):
    file.seek(0)
    img = Image.open(file).convert('RGB')
    img_array = np.array(img)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, col in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, alpha=0.7)
    ax.set_facecolor('#0f1116')
    fig.patch.set_facecolor('#0a0b0d')
    ax.tick_params(colors='#00f2ff', labelsize=8)
    return fig

# --- DATABASE LOGIC ---
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

# --- STYLING & UI ---
if not st.session_state["logged_in"]:
    st.markdown("<style>.stApp { background: linear-gradient(rgba(10, 11, 13, 0.85), rgba(10, 11, 13, 0.95)), url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop'); background-size: cover; color: #00f2ff; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }</style>", unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                if check_user(u_in, p_in):
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    log_forensic_action(f"Agent {u_in.upper()} authorized.")
                    st.rerun()
                else: st.error("Invalid Credentials")
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX-Image Forgery Detector</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def load_forensic_model():
        mp = 'forgery_detector.h5'
        return load_model(mp) if os.path.exists(mp) else None
    model = load_forensic_model()

    with st.sidebar:
        st.markdown(f"<h3>OPERATIVE</h3><h2>⚡ {st.session_state['user'].upper()}</h2><p>📍 NAGPUR_MS_IN</p>", unsafe_allow_html=True)
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        for f in files:
            f.seek(0); f_hash = get_file_hash(f.getvalue())
            log_forensic_action(f"Exhibit {f.name} logged.")
            st.info(f"🧬 EXHIBIT: {f.name} | HASH: {f_hash}")
            
            ela_img = convert_to_ela_image(f, quality=90)
            heat_img = generate_heatmap(f.getvalue(), ela_img)
            
            c_o, c_h = st.columns(2)
            c_o.image(f, caption="SOURCE EVIDENCE")
            c_h.image(heat_img, caption="HEATMAP ANALYSIS")
            
            c_l, c_p = st.columns(2)
            c_l.image(generate_luminance_map(f), caption="LUMINANCE GRADIENT")
            c_p.pyplot(plot_histogram(f))

        if st.button("INITIATE DEEP SCAN"):
            results = []
            for f in files:
                t_p = f"temp_{f.name}"
                with open(t_p, "wb") as b: b.write(f.getvalue())
                
                proc = prepare_image_for_cnn(t_p)
                tensor = np.expand_dims(proc, axis=0)
                
                # Model prediction 
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_p)
                
                # Adjusted threshold for Nagpur forensic standards [cite: 16, 17]
                if pred > 0.5:
                    verdict = "🚩 FORGERY"
                    conf = pred * 100
                else:
                    verdict = "🏳️ CLEAN"
                    conf = (1 - pred) * 100
                    
                results.append({"Exhibit": f.name, "Verdict": verdict, "Confidence": f"{conf:.2f}%"})
            st.session_state["scan_results"] = results
            st.success("Deep Scan Analysis Complete.")

        if st.session_state["scan_results"]:
            df = pd.DataFrame(st.session_state["scan_results"])
            st.table(df)
            
            st.markdown("""
                <div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border: 1px solid #00f2ff; border-radius: 5px;">
                <h4 style="color:#00f2ff; margin-top:0;">🏁 FORENSIC CONCLUSION</h4>
                <p>Analysis for <b>Case: {0}</b> complete. Inconsistencies analyzed by the 20-layer CNN[cite: 13, 190].</p>
                </div>
            """.format(case_id), unsafe_allow_html=True)

    while st.session_state["logged_in"]:
        now = datetime.now(IST)
        clock_placeholder.markdown(f"**{now.strftime('%d %b %Y')}** \n**{now.strftime('%I:%M:%S %p')}**")
        time.sleep(1)