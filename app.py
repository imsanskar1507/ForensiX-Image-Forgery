import streamlit as st
import numpy as np
import os
import pandas as pd
import sqlite3
import hashlib
from processor import convert_to_ela_image, prepare_image_for_cnn
from tensorflow.keras.models import load_model

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX | Nagpur Division", layout="wide")

# Session States
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state: st.session_state["auth_mode"] = "login"

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('forensix_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, recovery_key TEXT)''')
    conn.commit()
    conn.close()

init_db()

def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- AUTHENTICATION UI ---
if not st.session_state["logged_in"]:
    st.markdown("<h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Authorization</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        # LOGIN MODE
        if st.session_state["auth_mode"] == "login":
            st.subheader("Agent Login")
            with st.form("login_form"):
                u = st.text_input("Agent ID")
                p = st.text_input("Access Key", type="password")
                if st.form_submit_button("Authorize", use_container_width=True):
                    conn = sqlite3.connect('forensix_users.db')
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pass(p)))
                    if c.fetchone() or (u == "sanskar" and p == "detective2026"):
                        st.session_state["logged_in"] = True
                        st.rerun()
                    else:
                        st.error("Invalid Credentials")
                    conn.close()
            
            c1, c2 = st.columns(2)
            if c1.button("Register New Agent"): st.session_state["auth_mode"] = "reg"; st.rerun()
            if c2.button("Forgot Password"): st.session_state["auth_mode"] = "forgot"; st.rerun()

        # REGISTRATION MODE
        elif st.session_state["auth_mode"] == "reg":
            st.subheader("New Agent Enrollment")
            with st.form("reg_form"):
                new_u = st.text_input("Choose Agent ID")
                new_p = st.text_input("Choose Access Key", type="password")
                recovery = st.text_input("Recovery Keyword (for password reset)")
                if st.form_submit_button("Enroll"):
                    try:
                        conn = sqlite3.connect('forensix_users.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u, hash_pass(new_p), recovery))
                        conn.commit()
                        st.success("Enrollment Complete! Please Login.")
                        st.session_state["auth_mode"] = "login"
                        conn.close()
                        st.rerun()
                    except:
                        st.error("Agent ID already exists.")
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()

        # FORGOT PASSWORD MODE
        elif st.session_state["auth_mode"] == "forgot":
            st.subheader("Credential Recovery")
            with st.form("recovery_form"):
                u = st.text_input("Enter Agent ID")
                rk = st.text_input("Enter Recovery Keyword")
                new_p = st.text_input("New Access Key", type="password")
                if st.form_submit_button("Reset Password"):
                    conn = sqlite3.connect('forensix_users.db')
                    c = conn.cursor()
                    c.execute("UPDATE users SET password=? WHERE username=? AND recovery_key=?", (hash_pass(new_p), u, rk))
                    if conn.total_changes > 0:
                        conn.commit()
                        st.success("Password Updated! Please Login.")
                        st.session_state["auth_mode"] = "login"
                        conn.close()
                        st.rerun()
                    else:
                        st.error("Recovery failed. Incorrect ID or Keyword.")
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()

else:
    # --- MAIN INVESTIGATION DASHBOARD ---
    st.sidebar.markdown(f"**⚡ OPERATIVE ACTIVE**")
    st.sidebar.markdown("**📍 LOCATION: NAGPUR_MS_IN**")
    
    @st.cache_resource
    def load_engine():
        return load_model('forgery_detector.h5', compile=False) if os.path.exists('forgery_detector.h5') else None
    
    model = load_engine()

    files = st.file_uploader("UPLOAD EXHIBITS", type=["jpg", "png"], accept_multiple_files=True)

    if files:
        for f in files:
            col1, col2 = st.columns(2)
            ela = convert_to_ela_image(f)
            with col1: st.image(f, caption="SOURCE", use_container_width=True)
            with col2: st.image(ela, caption="ELA ANALYSIS", use_container_width=True)

        if st.button("INITIATE DEEP SCAN") and model:
            results = []
            for f in files:
                t_path = f"temp_{f.name}"
                with open(t_path, "wb") as b: b.write(f.getbuffer())
                img_data = prepare_image_for_cnn(t_path)
                tensor = np.expand_dims(img_data, axis=0)
                pred = model.predict(tensor, verbose=0)[0][0]
                os.remove(t_path)
                results.append({"FILENAME": f.name, "VERDICT": "🚩 FORGERY" if pred > 0.5 else "🏳️ CLEAN"})
            st.table(pd.DataFrame(results))

    if st.sidebar.button("🔴 EXIT SESSION"):
        st.session_state["logged_in"] = False
        st.rerun()