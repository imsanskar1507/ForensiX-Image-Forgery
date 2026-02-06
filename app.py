import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
import sqlite3
import hashlib
import cv2
import io
import zipfile
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"

# --- HEATMAP GENERATOR ---
def generate_heatmap(original_img_bytes, ela_img):
    # Convert original bytes to OpenCV format
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Convert ELA PIL image to OpenCV format
    ela_cv = np.array(ela_img.convert('RGB'))
    gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)
    
    # Create Heatmap
    heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay (60% Original, 40% Heatmap)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    return overlay

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    
    # Auto-repair schema if missing column
    c.execute("PRAGMA table_info(users)")
    if 'recovery' not in [col[1] for col in c.fetchall()]:
        c.execute('ALTER TABLE users ADD COLUMN recovery TEXT')
        
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit()
    conn.close()

def add_user(u, p, r):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        hp = hashlib.sha256(p.encode()).hexdigest()
        hr = hashlib.sha256(r.lower().strip().encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (u.lower().strip(), hp, hr))
        conn.commit()
        conn.close()
        return True
    except: return False

def check_user(u, p):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hp = hashlib.sha256(p.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hp))
    res = c.fetchone()
    conn.close()
    return res

def reset_password(u, r, npw):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hr = hashlib.sha256(r.lower().strip().encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND recovery=?", (u.lower().strip(), hr))
    if c.fetchone():
        nhp = hashlib.sha256(npw.encode()).hexdigest()
        c.execute("UPDATE users SET password=? WHERE username=?", (nhp, u.lower().strip()))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

init_db()

# --- CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', Courier, monospace; }
    .login-box { max-width: 450px; margin: auto; padding: 40px; background: #0f1116; border: 2px solid #00f2ff; box-shadow: 0px 0px 25px rgba(0, 242, 255, 0.2); border-radius: 10px; }
    .evidence-card { background: #0f1116; border: 1px solid #00f2ff; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center; }
    h1, h2, h3, h4 { color: #00f2ff !important; text-shadow: 0px 0px 8px #00f2ff; text-transform: uppercase; }
    .stButton>button { width: 100%; background: transparent; color: #00f2ff; border: 2px solid #00f2ff; font-weight: bold; }
    .stButton>button:hover { background: #00f2ff; color: black; box-shadow: 0px 0px 20px #00f2ff; }
    input { background-color: #050607 !important; color: #00f2ff !important; border: 1px solid #00f2ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔒 PORTAL ACCESS</h2>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["LOGIN", "REGISTER", "FORGOT KEY"])
        with t1:
            u_in = st.text_input("AGENT ID", key="l_u")
            p_in = st.text_input("ACCESS KEY", type="password", key="l_p")
            if st.button("AUTHORIZE"):
                if check_user(u_in, p_in):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u_in.strip()
                    st.rerun()
                else: st.error("INVALID CREDENTIALS")
        with t2:
            nu = st.text_input("NEW AGENT ID", key="r_u")
            npw = st.text_input("SET ACCESS KEY", type="password", key="r_p")
            cpw = st.text_input("CONFIRM KEY", type="password", key="r_cp")
            rec = st.text_input("SECRET RECOVERY WORD", key="r_rec")
            if st.button("CREATE ACCOUNT"):
                if npw != cpw: st.error("Keys do not match")
                elif len(npw) < 6: st.error("Key too short")
                elif add_user(nu, npw, rec): st.success("Registered. Use Login.")
                else: st.error("ID already exists")
        with t3:
            st.markdown("### 🔑 KEY RECOVERY")
            fu = st.text_input("TARGET AGENT ID", key="f_u")
            frec = st.text_input("SECRET WORD", type="password", key="f_rec")
            fnpw = st.text_input("NEW ACCESS KEY", type="password", key="f_npw")
            if st.button("RESET ACCESS"):
                if reset_password(fu, frec, fnpw): st.success("Key updated.")
                else: st.error("Verification Failed")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    @st.cache_resource
    def get_model():
        bp = os.path.dirname(__file__)
        mp = os.path.join(bp, 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None

    model = get_model()

    with st.sidebar:
        agent = st.session_state.get('user', 'Agent')
        st.markdown(f"### ⚡ ACTIVE: {agent.upper()}")
        if st.button("TERMINATE SESSION"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        case_notes = st.text_area("INVESTIGATOR NOTES", height=150)

    st.markdown("<h1>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    
    files = st.file_uploader("SUBMIT DIGITAL EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        # If single file, show the advanced 3-column forensic view
        if len(files) == 1:
            file = files[0]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="evidence-card"><h4>SOURCE</h4>', unsafe_allow_html=True)
                st.image(file, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="evidence-card"><h4>ELA MAP</h4>', unsafe_allow_html=True)
                ela_img = convert_to_ela_image(file, quality=90)
                st.image(ela_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="evidence-card"><h4>HEATMAP</h4>', unsafe_allow_html=True)
                heatmap_res = generate_heatmap(file.getvalue(), ela_img)
                st.image(heatmap_res, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("INITIATE DEEP SCAN INTERROGATION"):
            results = []
            bar = st.progress(0)
            with st.status("📡 ANALYZING DATASTREAMS...", expanded=True) as status:
                for idx, f in enumerate(files):
                    tmp = f"temp_{f.name}"
                    with open(tmp, "wb") as buffer: buffer.write(f.getbuffer())
                    
                    meta_data, meta_msg = scan_metadata(tmp)
                    proc = prepare_image_for_cnn(tmp)
                    pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                    os.remove(tmp)
                    
                    verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                    results.append({
                        "FILENAME": f.name, "VERDICT": verdict, 
                        "CONFIDENCE": float(max(pred, 1-pred)*100), "METADATA": meta_msg
                    })
                    bar.progress((idx + 1) / len(files))
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