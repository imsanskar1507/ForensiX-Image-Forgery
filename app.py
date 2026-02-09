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

# --- CORE UTILITIES ---
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def generate_heatmap(original_img_bytes, ela_img):
    nparr = np.frombuffer(original_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ela_cv = np.array(ela_img.convert('RGB'))
    gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)
    heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

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
    return c.fetchall()

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
        hp, hr = hashlib.sha256(p.encode()).hexdigest(), hashlib.sha256(r.lower().strip().encode()).hexdigest()
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

# --- DYNAMIC CSS ENGINE ---
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
            border: 2px solid #00f2ff; border-radius: 15px; padding: 25px; box-shadow: 0px 0px 30px rgba(0, 242, 255, 0.2);
        }
        h1, h2 { color: #00f2ff !important; text-shadow: 0px 0px 15px rgba(0, 242, 255, 0.8); }
        .stButton>button { background: transparent; color: #00f2ff; border: 2px solid #00f2ff; }
        .stButton>button:hover { background: #00f2ff; color: #000; box-shadow: 0px 0px 25px #00f2ff; }
        input { background-color: rgba(0, 0, 0, 0.5) !important; color: #00f2ff !important; border: 1px solid #00f2ff !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
        .evidence-card {
            background: #0f1116; border: 1px solid #00f2ff; border-radius: 12px;
            padding: 20px; margin-bottom: 20px; box-shadow: 0px 0px 15px rgba(0, 242, 255, 0.1);
        }
        .dossier-header {
            background-color: #00f2ff; color: #000; padding: 5px 15px; font-weight: bold;
            font-size: 12px; border-radius: 5px 5px 0 0; letter-spacing: 1px; display: inline-block;
        }
        .dossier-box {
            background: rgba(0, 242, 255, 0.05) !important; border: 1px solid #00f2ff !important;
            border-radius: 0 5px 5px 5px; padding: 10px; margin-bottom: 20px;
        }
        section[data-testid="stSidebar"] .stTextArea textarea {
            background-color: transparent !important; border: none !important; color: #00f2ff !important;
            font-family: 'Courier New', Courier, monospace !important; font-size: 13px !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'>🔒 PORTAL ACCESS</h2>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["LOGIN", "REGISTER", "FORGOT KEY"])
        with t1:
            u_in, p_in = st.text_input("AGENT ID", key="l_u"), st.text_input("ACCESS KEY", type="password", key="l_p")
            if st.button("AUTHORIZE"):
                if check_user(u_in, p_in):
                    st.session_state["logged_in"], st.session_state["user"] = True, u_in.strip()
                    st.rerun()
                else: st.error("INVALID CREDENTIALS")
        with t2:
            nu, npw, ncpw, rec = st.text_input("NEW ID"), st.text_input("KEY", type="password"), st.text_input("CONFIRM", type="password"), st.text_input("RECOVERY WORD")
            if st.button("CREATE ACCOUNT"):
                if npw != ncpw: st.error("Keys mismatch")
                elif add_user(nu, npw, rec): st.success("Created. Proceed to Login.")
                else: st.error("ID exists")
        with t3:
            fu, frec, fnpw = st.text_input("ID", key="f_u"), st.text_input("SECRET", type="password"), st.text_input("NEW KEY", type="password")
            if st.button("RESET"):
                if reset_password(fu, frec, fnpw): st.success("Updated.")
                else: st.error("Failed")
        st.markdown('</div>', unsafe_allow_html=True)

else:
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
                <p style="margin:5px 0 0 0; font-size: 10px; color: #00f2ff;">LOCATION: NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)
        case_id = st.text_input("CASE ID", value="REF-ALPHA-01")
        st.markdown('<div class="dossier-header">📝 INVESTIGATION LOG</div><div class="dossier-box">', unsafe_allow_html=True)
        case_notes = st.text_area("FIELD NOTES", height=250, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🔴 EXIT SYSTEM"):
            st.session_state["logged_in"] = False; st.rerun()

    st.markdown("<h1>🛰️ ForensiX-Image Forgery Detector</h1>", unsafe_allow_html=True)
    
    if st.session_state["user"].lower() == "sanskar":
        tab_main, tab_admin = st.tabs(["🔍 INVESTIGATION", "📊 ADMIN CONSOLE"])
    else: tab_main, tab_admin = st.container(), None

    with tab_main:
        files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
        if files:
            for f in files:
                f_hash = get_file_hash(f.getvalue())
                st.info(f"🧬 EXHIBIT {f.name} Fingerprint: {f_hash}")
                ela_img = convert_to_ela_image(f, quality=90)
                heat_img = generate_heatmap(f.getvalue(), ela_img)
                c_o, c_h = st.columns(2)
                with c_o: 
                    st.markdown('<div class="evidence-card"><h4>SOURCE</h4>', unsafe_allow_html=True)
                    st.image(f, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
                with c_h: 
                    st.markdown('<div class="evidence-card"><h4>HEATMAP</h4>', unsafe_allow_html=True)
                    st.image(heat_img, use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

            if st.button("INITIATE DEEP SCAN"):
                results, zip_buffer = [], io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
                    bar = st.progress(0)
                    with st.status("📡 ANALYZING...", expanded=True) as status:
                        for idx, f in enumerate(files):
                            tmp = f"temp_{f.name}"
                            with open(tmp, "wb") as b: b.write(f.getbuffer())
                            _, m_msg = scan_metadata(tmp)
                            proc = prepare_image_for_cnn(tmp)
                            pred = model.predict(np.expand_dims(proc, axis=0))[0][0]
                            ela = convert_to_ela_image(f, quality=90)
                            heat = generate_heatmap(f.getvalue(), ela)
                            zf.writestr(f"Source_{f.name}", f.getvalue())
                            e_io = io.BytesIO(); ela.save(e_io, format="PNG"); zf.writestr(f"ELA_{f.name}.png", e_io.getvalue())
                            h_io = io.BytesIO(); Image.fromarray(heat).save(h_io, format="PNG"); zf.writestr(f"Heatmap_{f.name}.png", h_io.getvalue())
                            os.remove(tmp)
                            v = "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"
                            results.append({"FILENAME": f.name, "VERDICT": v, "CONFIDENCE": float(max(pred, 1-pred)*100), "METADATA": m_msg})
                            bar.progress((idx+1)/len(files))
                        pdf_d = create_pdf_report(results, case_notes=case_notes)
                        zf.writestr(f"Report_{case_id}.pdf", pdf_d)
                        status.update(label="COMPLETE", state="complete")
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("SCANNED", len(results)); c2.metric("FLAGGED", len(df[df['VERDICT'] == "🚩 FORGERY"]))
                with c3: st.download_button("📥 EXPORT BUNDLE (.ZIP)", zip_buffer.getvalue(), f"CASE_{case_id}.zip")
                st.markdown('</div>', unsafe_allow_html=True)

    if tab_admin:
        with tab_admin:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="evidence-card"><h4>AGENTS</h4>', unsafe_allow_html=True)
                u_list = get_all_users()
                for u_e in u_list:
                    u_n = u_e[0]
                    ca1, ca2 = st.columns([3, 1])
                    ca1.text(f"👤 {u_n.upper()}")
                    if u_n != "sanskar" and ca2.button("REVOKE", key=f"d_{u_n}"): 
                        if delete_user(u_n): st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="evidence-card"><h4>METRICS</h4>', unsafe_allow_html=True)
                st.metric("DB STATUS", "ONLINE"); st.metric("AGENTS", len(u_list))
                st.markdown('</div>', unsafe_allow_html=True)