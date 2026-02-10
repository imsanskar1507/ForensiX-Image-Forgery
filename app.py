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

def generate_luminance_map(file):
    img = Image.open(file).convert('L')
    img_array = np.array(img, dtype=float)
    dx, dy = np.gradient(img_array)
    gradient = np.sqrt(dx**2 + dy**2)
    gradient = (gradient / (gradient.max() if gradient.max() > 0 else 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)

def plot_histogram(file):
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
        .dossier-box { background: rgba(25, 27, 32, 0.95) !important; border: 1px solid #00f2ff !important; border-radius: 0 5px 5px 5px; padding: 10px; }
        </style>
        """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
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
            if c2.button("Forgot"): st.session_state["auth_mode"] = "forgot"; st.rerun()
        # (Registration and Forgot logic remains same as your original...)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = os.path.join(os.path.dirname(__file__), 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None
    
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ AGENT: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("FIELD NOTES", height=150)
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    st.markdown("---")
    
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        for f in files:
            f_hash = get_file_hash(f.getvalue())
            st.info(f"🧬 EXHIBIT {f.name} | HASH: {f_hash}")
            c_o, c_h = st.columns(2)
            ela_img = convert_to_ela_image(f, quality=90)
            heat_img = generate_heatmap(f.getvalue(), ela_img)
            with c_o: st.image(f, caption="SOURCE EVIDENCE")
            with c_h: st.image(heat_img, caption="HEATMAP ANALYSIS")
            
            c_l, c_p = st.columns(2)
            with c_l: st.image(generate_luminance_map(f), caption="LUMINANCE GRADIENT")
            with c_p: st.pyplot(plot_histogram(f))

        # --- REPAIRED DEEP SCAN LOGIC ---
        if st.button("INITIATE DEEP SCAN"):
            results, zip_buffer = [], io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 ANALYZING...", expanded=True) as status:
                    for idx, f in enumerate(files):
                        # 1. Save temp file for metadata/processing
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        # 2. Run Metadata Scan
                        has_meta, m_msg = scan_metadata(tmp)
                        
                        # 3. Run CNN Prediction
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        
                        # 4. Store Results
                        v = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                        conf = float(max(pred, 1-pred)*100)
                        results.append({
                            "FILENAME": f.name, 
                            "VERDICT": v, 
                            "CONFIDENCE": f"{conf:.2f}%", 
                            "METADATA": "DETECTED" if has_meta else "NONE"
                        })
                        bar.progress((idx+1)/len(files))
                    
                    # 5. Generate PDF Report
                    pdf_data = create_pdf_report(results, case_notes=case_notes)
                    zf.writestr(f"Forensic_Report_{case_id}.pdf", pdf_data)
                    status.update(label="SCAN COMPLETE", state="complete")
            
            # 6. DISPLAY ANALYZED DATA TABLE
            st.markdown("### 📊 FINAL DETERMINATION SUMMARY")
            st.table(pd.DataFrame(results))
            st.download_button("📥 DOWNLOAD CASE DOSSIER (.ZIP)", zip_buffer.getvalue(), f"CASE_{case_id}.zip")

    # --- CLOCK REFRESH ---
    while st.session_state["logged_in"]:
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
                <span style="color: #00f2ff; font-size: 16px; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 24px; font-family: 'Courier New';">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)