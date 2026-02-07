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

# Initialize Session States
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"

# --- CORE UTILITIES ---
def generate_heatmap(original_img_bytes, ela_img):
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ela_cv = np.array(ela_img.convert('RGB'))
    gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)
    heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    return overlay

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
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

def get_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = c.fetchall()
    conn.close()
    return users

def delete_user(username):
    if username.lower() == "sanskar": return False
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return True

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

# --- THEMED CSS (Neon Data Grid) ---
st.markdown("""
    <style>
    /* Background Image with Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(10, 11, 13, 0.7), rgba(10, 11, 13, 0.85)), 
                    url("https://www.dreamstime.com/abstract-futuristic-background-binary-code-digital-data-screen-dark-concept-image211438962");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #00f2ff;
        font-family: 'Courier New', Courier, monospace;
    }

    /* Glassmorphism Evidence Cards */
    .evidence-card, .login-box {
        background: rgba(15, 17, 22, 0.7) !important;
        backdrop-filter: blur(12px);
        border: 2px solid rgba(0, 242, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0px 8px 32px 0 rgba(0, 242, 255, 0.2);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(0, 242, 255, 0.05);
        border-radius: 5px 5px 0px 0px;
        color: #00f2ff;
    }
    
    /* Neon Text Shadows */
    h1, h2, h3, h4 { 
        color: #00f2ff !important; 
        text-shadow: 0px 0px 15px rgba(0, 242, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Tactical Buttons */
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background-color: #00f2ff;
        color: #000;
        box-shadow: 0px 0px 25px #00f2ff;
        transform: translateY(-2px);
    }

    /* Inputs */
    input { 
        background-color: rgba(5, 6, 7, 0.6) !important; 
        color: #00f2ff !important; 
        border: 1px solid rgba(0, 242, 255, 0.4) !important; 
    }
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
            st.markdown("### 🔑 RECOVERY PROTOCOL")
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
    
    if st.session_state["user"].lower() == "sanskar":
        tab_main, tab_admin = st.tabs(["🔍 INVESTIGATION", "📊 ADMIN CONSOLE"])
    else:
        tab_main = st.container()
        tab_admin = None

    with tab_main:
        files = st.file_uploader("SUBMIT DIGITAL EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)

        if files:
            st.markdown("### 🧬 FORENSIC SIDE-BY-SIDE")
            for f in files:
                ela_img = convert_to_ela_image(f, quality=90)
                heatmap_img = generate_heatmap(f.getvalue(), ela_img)
                
                col_orig, col_comp = st.columns(2)
                with col_orig:
                    st.markdown('<div class="evidence-card"><h4>ORIGINAL SOURCE</h4>', unsafe_allow_html=True)
                    st.image(f, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_comp:
                    st.markdown('<div class="evidence-card"><h4>TAMPER HEATMAP</h4>', unsafe_allow_html=True)
                    st.image(heatmap_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if st.button("INITIATE DEEP SCAN INTERROGATION"):
                results = []
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    progress_bar = st.progress(0)
                    with st.status("📡 ANALYZING SCENE...", expanded=True) as status:
                        for idx, f in enumerate(files):
                            tmp = f"temp_{f.name}"
                            with open(tmp, "wb") as buffer: buffer.write(f.getbuffer())
                            
                            meta_data, meta_msg = scan_metadata(tmp)
                            proc = prepare_image_for_cnn(tmp)
                            pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                            
                            ela_asset = convert_to_ela_image(f, quality=90)
                            heatmap_asset = generate_heatmap(f.getvalue(), ela_asset)
                            
                            zip_file.writestr(f"Source_{f.name}", f.getvalue())
                            ela_io = io.BytesIO(); ela_asset.save(ela_io, format="PNG")
                            zip_file.writestr(f"ELA_{f.name}.png", ela_io.getvalue())
                            heat_io = io.BytesIO(); Image.fromarray(heatmap_asset).save(heat_io, format="PNG")
                            zip_file.writestr(f"Heatmap_{f.name}.png", heat_io.getvalue())
                            
                            os.remove(tmp)
                            verdict = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                            results.append({"FILENAME": f.name, "VERDICT": verdict, "CONFIDENCE": float(max(pred, 1-pred)*100), "METADATA": meta_msg})
                            progress_bar.progress((idx + 1) / len(files))
                        
                        pdf_data = create_pdf_report(results, case_notes=case_notes)
                        zip_file.writestr(f"Forensic_Report_{case_id}.pdf", pdf_data)
                        status.update(label="SCAN COMPLETE", state="complete")

                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("SCANNED", len(results))
                c2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]))
                with c3:
                    st.download_button("📥 EXPORT TACTICAL BUNDLE (.ZIP)", zip_buffer.getvalue(), f"CASE_{case_id}.zip", "application/zip")
                st.markdown('</div>', unsafe_allow_html=True)

    if tab_admin:
        with tab_admin:
            st.markdown("### 🛠️ SYSTEM ADMINISTRATION")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="evidence-card"><h4>REGISTERED AGENTS</h4>', unsafe_allow_html=True)
                user_list = get_all_users()
                for u_entry in user_list:
                    u_name = u_entry[0]
                    c_u1, c_u2 = st.columns([3, 1])
                    c_u1.text(f"👤 {u_name.upper()}")
                    if u_name != "sanskar":
                        if c_u2.button("REVOKE", key=f"del_{u_name}"):
                            if delete_user(u_name): st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="evidence-card"><h4>METRICS</h4>', unsafe_allow_html=True)
                st.metric("DATABASE STATUS", "ONLINE")
                st.metric("TOTAL AGENTS", len(user_list))
                st.markdown('</div>', unsafe_allow_html=True)