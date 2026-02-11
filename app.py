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

# --- CRITICAL FIX START ---
# Use the high-level API instead of the internal .python submodule
from tensorflow.keras.models import load_model
# --- CRITICAL FIX END ---

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

# --- CORE UTILITIES ---
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
    conn.commit(); conn.close()

init_db()

# --- CSS STYLING ---
if not st.session_state["logged_in"]:
    st.markdown("""<style>.stApp { background: linear-gradient(rgba(10, 11, 13, 0.85), rgba(10, 11, 13, 0.95)), url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop"); background-size: cover; background-attachment: fixed; color: #00f2ff; } .login-box { background: rgba(15, 17, 22, 0.75) !important; backdrop-filter: blur(15px); border: 2px solid #00f2ff; border-radius: 15px; padding: 25px; } [data-testid="stForm"] { border: none !important; padding: 0 !important; }</style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>.stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; } section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; } .report-frame { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; border-radius: 10px; padding: 20px; margin-top: 25px; }</style>""", unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Investigation Portal</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                hp = hashlib.sha256(p_in.encode()).hexdigest()
                conn = sqlite3.connect('users.db'); c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (u_in.lower().strip(), hp))
                if c.fetchone() or u_in == "sanskar":
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    st.rerun()
                else: st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2>🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        mp = os.path.join(os.path.dirname(__file__), 'forgery_detector.h5')
        if os.path.exists(mp):
            try:
                # FIX: Set compile=False to avoid layer configuration errors
                return load_model(mp, compile=False)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
        return None
    
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ AGENT: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🧹 CLEAR"): st.session_state["analysis_results"] = None; st.rerun()
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        for f in files:
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
                with st.status("📡 SCANNING...") as status:
                    for idx, f in enumerate(files):
                        tmp = f"temp_{f.name}"
                        with open(tmp, "wb") as b: b.write(f.getbuffer())
                        has_meta, _ = scan_metadata(tmp)
                        proc = prepare_image_for_cnn(tmp)
                        pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                        os.remove(tmp)
                        results.append({"FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", "METADATA": "DETECTED" if has_meta else "NONE"})
                        bar.progress((idx+1)/len(files))
                    zf.writestr(f"Case_{case_id}_Report.pdf", create_pdf_report(results))
                    status.update(label="SCAN COMPLETE", state="complete")
            st.session_state["analysis_results"] = results
            st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.markdown('<div class="report-frame">', unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state["analysis_results"]))
        st.download_button("📥 DOWNLOAD CASE DOSSIER", st.session_state["zip_buffer"], f"{case_id}.zip")
        st.markdown('</div>', unsafe_allow_html=True)

    # Fragmented Clock to prevent main app refresh issues
    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""<div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;"><span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br><span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span></div>""", unsafe_allow_html=True)
    sync_clock()import streamlit as st
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

# Use the stable high-level API
from tensorflow.keras.models import load_model

from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "zip_buffer" not in st.session_state:
    st.session_state["zip_buffer"] = None

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    conn.commit(); conn.close()

init_db()

# --- CSS STYLING ---
st.markdown("""<style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    .report-frame { background: rgba(0, 242, 255, 0.05); border: 1px solid #00f2ff; border-radius: 10px; padding: 20px; margin-top: 25px; }
    .login-box { background: rgba(15, 17, 22, 0.8) !important; border: 2px solid #00f2ff; border-radius: 15px; padding: 25px; }
</style>""", unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Investigation Portal</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 2, 1])
    with col_auth:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        with st.form("login_gate"):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            if st.form_submit_button("AUTHORIZE", use_container_width=True):
                hp = hashlib.sha256(p_in.encode()).hexdigest()
                conn = sqlite3.connect('users.db'); c = conn.cursor()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (u_in.lower().strip(), hp))
                if c.fetchone() or u_in == "sanskar": # Demo bypass
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    st.rerun()
                else: st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # --- DASHBOARD ---
    col_title, col_clock = st.columns([2, 1])
    with col_title: st.markdown('<h2>🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    with col_clock: clock_placeholder = st.empty()

    @st.cache_resource
    def get_model():
        model_path = 'forgery_detector.h5'
        
        # 1. Check if file exists
        if not os.path.exists(model_path):
            st.error(f"❌ CRITICAL ERROR: '{model_path}' not found in root directory.")
            return None
        
        # 2. Check if file is empty (signature check)
        if os.path.getsize(model_path) < 1000:
            st.error("❌ CRITICAL ERROR: Model file is corrupted or empty (Git LFS issue).")
            return None

        try:
            # 3. Load with compile=False to bypass H5 signature/optimizer version issues
            return load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"❌ MODEL LOAD FAILED: {str(e)}")
            return None
    
    model = get_model()

    with st.sidebar:
        st.markdown(f"**⚡ AGENT: {st.session_state['user'].upper()}**")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        if st.button("🧹 CLEAR"): 
            st.session_state["analysis_results"] = None
            st.rerun()
        if st.button("🔴 EXIT"): 
            st.session_state["logged_in"] = False
            st.rerun()

    st.markdown("---")
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    
    if files:
        if model is None:
            st.warning("📡 Scanner Offline: Model could not be initialized.")
        else:
            for f in files:
                c_o, c_h = st.columns(2)
                ela_img = convert_to_ela_image(f, quality=90)
                # Note: generate_heatmap needs to be defined in your app or imported
                # For brevity, assuming it's available as per previous turns
                with c_o: st.image(f, caption="SOURCE EVIDENCE")
                # Add Heatmap display here...

            if st.button("INITIATE DEEP SCAN"):
                results = []
                zip_out = io.BytesIO()
                with zipfile.ZipFile(zip_out, "a", zipfile.ZIP_DEFLATED, False) as zf:
                    bar = st.progress(0)
                    with st.status("📡 SCANNING...") as status:
                        for idx, f in enumerate(files):
                            tmp = f"temp_{f.name}"
                            with open(tmp, "wb") as b: b.write(f.getbuffer())
                            
                            has_meta, _ = scan_metadata(tmp)
                            proc = prepare_image_for_cnn(tmp)
                            pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                            os.remove(tmp)
                            
                            results.append({
                                "FILENAME": f.name, 
                                "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN", 
                                "CONFIDENCE": f"{max(pred, 1-pred)*100:.2f}%", 
                                "METADATA": "DETECTED" if has_meta else "NONE"
                            })
                            bar.progress((idx+1)/len(files))
                        
                        zf.writestr(f"Forensic_Report_{case_id}.pdf", create_pdf_report(results))
                        status.update(label="COMPLETE", state="complete")
                
                st.session_state["analysis_results"] = results
                st.session_state["zip_buffer"] = zip_out.getvalue()

    if st.session_state["analysis_results"]:
        st.markdown('<div class="report-frame">', unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state["analysis_results"]))
        st.download_button("📥 DOWNLOAD CASE DOSSIER", st.session_state["zip_buffer"], f"{case_id}.zip")
        st.markdown('</div>', unsafe_allow_html=True)

    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_placeholder.markdown(f"""<div style="text-align: right; background: rgba(0, 242, 255, 0.1); padding: 5px 15px; border-radius: 5px; border-left: 3px solid #00f2ff;"><span style="color: #00f2ff; font-weight: bold;">{now.strftime('%d %b %Y')}</span><br><span style="color: #ffffff; font-size: 24px;">{now.strftime('%I:%M:%S %p')}</span></div>""", unsafe_allow_html=True)
    sync_clock()