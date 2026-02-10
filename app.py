import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pytz  # Handles the Time Zone sync
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
from tensorflow.python.keras.models import load_model
from report_gen import create_pdf_report 

# --- NEW UTILITY FOR AI DETECTION ---
def detect_ai_generated_patterns(image_path):
    """Analyses frequency artifacts typical of GAN/Diffusion generated images."""
    try:
        img = cv2.imread(image_path, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Calculate variance in high-frequency regions
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        # Mask center (low frequency)
        magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30] = 0
        score = np.var(magnitude_spectrum)
        
        if score > 120: return f"{min(98.9, score/1.5):.1f}%", "High (Synthetic)"
        elif score > 70: return f"{score/1.8:.1f}%", "Possible AI"
        else: return f"{score/4:.1f}%", "Low (Natural)"
    except:
        return "0.0%", "Scan Error"

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")

# LOCK TIME TO INDIA STANDARD TIME (IST)
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"  # Options: login, register, forgot
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state:
    st.session_state["case_log"] = []
# Persistent states for the Report Section
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state:
    st.session_state["zip_buffer"] = None

# --- CORE UTILITIES ---
def get_timestamp():
    """Returns the current IST time for forensic logging."""
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
        [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
        .dossier-box {
            background: rgba(25, 27, 32, 0.95) !important;
            border: 1px solid #00f2ff !important;
            border-radius: 0 5px 5px 5px; padding: 10px;
        }
        /* Custom Report Section Style */
        .report-frame {
            background: rgba(0, 242, 255, 0.05);
            border: 1px solid #00f2ff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
        }
        </style>
        """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # --- LOGIN MODE ---
        if st.session_state["auth_mode"] == "login":
            with st.form("login_gate"):
                u_in = st.text_input("AGENT ID")
                p_in = st.text_input("ACCESS KEY", type="password")
                submitted = st.form_submit_button("AUTHORIZE", use_container_width=True)
                if submitted:
                    if check_user(u_in, p_in):
                        st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                        log_forensic_action(f"Agent {u_in.upper()} authorized.")
                        st.rerun()
                    else:
                        st.error("Invalid Credentials")
            
            # Nav links
            c1, c2 = st.columns(2)
            if c1.button("New Registration", use_container_width=True):
                st.session_state["auth_mode"] = "register"
                st.rerun()
            if c2.button("Forgot Password", use_container_width=True):
                st.session_state["auth_mode"] = "forgot"
                st.rerun()

        # --- REGISTRATION MODE ---
        elif st.session_state["auth_mode"] == "register":
            with st.form("register_gate"):
                st.markdown("### Agent Enrollment")
                new_u = st.text_input("SET AGENT ID")
                new_p = st.text_input("SET ACCESS KEY", type="password")
                new_r = st.text_input("RECOVERY HINT (e.g. Nagpur)")
                reg_submit = st.form_submit_button("ENROLL AGENT", use_container_width=True)
                if reg_submit:
                    if new_u and new_p:
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        try:
                            hp = hashlib.sha256(new_p.encode()).hexdigest()
                            hr = hashlib.sha256(new_r.encode()).hexdigest()
                            c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u.lower().strip(), hp, hr))
                            conn.commit()
                            st.success("Registration Successful! Please Login.")
                            st.session_state["auth_mode"] = "login"
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("Agent ID already exists.")
                        finally:
                            conn.close()
            if st.button("Back to Login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()

        # --- FORGOT PASSWORD MODE ---
        elif st.session_state["auth_mode"] == "forgot":
            with st.form("forgot_gate"):
                st.markdown("### Credential Recovery")
                f_u = st.text_input("AGENT ID")
                f_r = st.text_input("RECOVERY HINT")
                f_np = st.text_input("NEW ACCESS KEY", type="password")
                reset_submit = st.form_submit_button("RESET ACCESS KEY", use_container_width=True)
                if reset_submit:
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    hr = hashlib.sha256(f_r.encode()).hexdigest()
                    c.execute("SELECT * FROM users WHERE username=? AND recovery=?", (f_u.lower().strip(), hr))
                    if c.fetchone():
                        hp = hashlib.sha256(f_np.encode()).hexdigest()
                        c.execute("UPDATE users SET password=? WHERE username=?", (hp, f_u.lower().strip()))
                        conn.commit()
                        st.success("Password Reset Successful!")
                        st.session_state["auth_mode"] = "login"
                        st.rerun()
                    else:
                        st.error("Recovery hint mismatch.")
                    conn.close()
            if st.button("Back to Login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
else:
    # --- NAV BAR WITH LARGE AUTOMATED IST CLOCK ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin:0; color:#00f2ff;">🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    
    with col_clock:
        clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = os.path.join(os.path.dirname(__file__), 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None
    
    model = get_model()

    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 25px;">
                <h4 style="margin:0; font-size: 14px; opacity: 0.8;">OPERATIVE STATUS</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 22px;">⚡ {st.session_state['user'].upper()}</h2>
                <p style="margin:10px 0 0 0; font-size: 14px; color: #00f2ff; font-weight: bold;">📍 LOCATION: NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📜 SESSION LOG")
        with st.expander("Chain of Custody", expanded=False):
            for entry in st.session_state["case_log"]: st.text(entry)

        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("FIELD NOTES", height=150)
        
        if st.button("🧹 CLEAR RESULTS"):
            st.session_state["analysis_results"] = None
            st.session_state["zip_buffer"] = None
            st.rerun()
            
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

        if st.button("INITIATE DEEP SCAN"):
            results = []
            zip_out = io.BytesIO()
            with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                bar = st.progress(0)
                with st.status("📡 ANALYZING...", expanded=True) as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        
                        # --- INTEGRATED AI/DEEPFAKE SCAN ---
                        ai_conf, ai_risk = detect_ai_generated_patterns(tmp)
                        
                        has_meta, m_msg = scan_metadata(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        
                        v = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                        results.append({
                            "FILENAME": f.name, 
                            "FORGERY": v, 
                            "CNN CONF.": f"{max(pred, 1-pred)*100:.2f}%", 
                            "AI/DEEPFAKE": ai_conf,
                            "RISK": ai_risk,
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

    # --- THE LIVE CLOCK REFRESH LOOP ---
    while st.session_state["logged_in"]:
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;">
                <span style="color: #00f2ff; font-size: 16px; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br>
                <span style="color: #ffffff; font-size: 24px; font-family: 'Courier New';">{now.strftime('%I:%M:%S %p')}</span>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)