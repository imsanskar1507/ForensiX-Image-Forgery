import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
import sqlite3
import hashlib
from processor import convert_to_ela_image, prepare_image_for_cnn
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Tactical Suite", layout="wide", page_icon="🕵️")

# Initialize Session States
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    # Default admin account
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hashed_pw = hashlib.sha256("detective2026".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?)", ("sanskar", hashed_pw))
    conn.commit()
    conn.close()

def add_user(username, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?)", (username.lower().strip(), hashed_pw))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username.lower().strip(), hashed_pw))
    result = c.fetchone()
    conn.close()
    return result

init_db()

# --- CYBER CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', Courier, monospace; }
    .login-box {
        max-width: 450px; margin: auto; padding: 40px;
        background: #0f1116; border: 2px solid #00f2ff;
        box-shadow: 0px 0px 25px rgba(0, 242, 255, 0.2); border-radius: 10px;
    }
    .evidence-card {
        background: #0f1116; border: 1px solid #00f2ff;
        border-radius: 8px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0px 0px 10px rgba(0, 242, 255, 0.1);
    }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 0px 0px 8px #00f2ff; }
    .stButton>button {
        width: 100%; background: transparent; color: #00f2ff;
        border: 2px solid #00f2ff; font-weight: bold;
    }
    .stButton>button:hover { background: #00f2ff; color: black; box-shadow: 0px 0px 20px #00f2ff; }
    input { background-color: #050607 !important; color: #00f2ff !important; border: 1px solid #00f2ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔒 PORTAL ACCESS</h2>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["LOGIN", "REGISTER"])
        
        with t1:
            u = st.text_input("AGENT ID", key="l_u")
            p = st.text_input("ACCESS KEY", type="password", key="l_p")
            if st.button("AUTHORIZE"):
                if check_user(u, p):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u.strip()
                    st.success("ACCESS GRANTED")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("INVALID CREDENTIALS")
        
        with t2:
            nu = st.text_input("NEW AGENT ID", key="r_u")
            npw = st.text_input("SET ACCESS KEY", type="password", key="r_p")
            if st.button("CREATE ACCOUNT"):
                if len(npw) < 6: st.error("Key too short")
                elif add_user(nu, npw): st.success("Account Created. Switch to Login tab.")
                else: st.error("ID already exists")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- MAIN FORENSIC SUITE ---
    @st.cache_resource
    def get_model():
        bp = os.path.dirname(__file__)
        mp = os.path.join(bp, 'forgery_detector.h5')
        return load_model(mp) if os.path.exists(mp) else None

    model = get_model()

    with st.sidebar:
        # FIXED: Use .get to avoid KeyError
        current_agent = st.session_state.get('user', 'Agent')
        st.markdown(f"### ⚡ ACTIVE: {current_agent.upper()}")
        if st.button("EXIT SYSTEM"):
            st.session_state["logged_in"] = False
            st.rerun()
        st.markdown("---")
        case_id = st.text_input("CASE ID", value="REF-X")
        case_notes = st.text_area("AGENT NOTES", height=150)

    st.markdown("<h1>🛰️ FORENSIX: TACTICAL SCANNER</h1>", unsafe_allow_html=True)
    
    files = st.file_uploader("SUBMIT EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        if st.button("INITIATE INTERROGATION"):
            results = []
            bar = st.progress(0)
            with st.status("📡 SCANNING...") as status:
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
                pdf = create_pdf_report(results, case_notes=case_notes)
                st.download_button("📥 DOWNLOAD DOSSIER", pdf, f"{case_id}.pdf")
            st.markdown('</div>', unsafe_allow_html=True)