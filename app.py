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
from metadata_scanner import scan_metadata
from tensorflow.keras.models import load_model
from report_gen import create_pdf_report 

# --- INITIAL CONFIG ---
st.set_page_config(page_title="ForensiX Detector", layout="wide", page_icon="🕵️")
IST = pytz.timezone('Asia/Kolkata')

# Session State Management
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"  # login, register, or forgot
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"

# --- DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)''')
    # Default Admin for G H Raisoni Demo
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        hr = hashlib.sha256("nagpur".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", ("sanskar", hp, hr))
    conn.commit()
    conn.close()

init_db()

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(10, 11, 13, 0.8), rgba(10, 11, 13, 0.9)), 
                    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-attachment: fixed; color: #00f2ff;
    }
    [data-testid="stForm"] { border: none !important; padding: 0 !important; }
    .login-box {
        background: rgba(15, 17, 22, 0.85) !important; 
        backdrop-filter: blur(10px);
        border: 2px solid #00f2ff; border-radius: 15px; padding: 35px;
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.3);
    }
    .auth-link { color: #00f2ff; cursor: pointer; text-decoration: underline; font-size: 13px; }
    </style>
    """, unsafe_allow_html=True)

# --- AUTHENTICATION HELPERS ---
def hash_val(val):
    return hashlib.sha256(val.encode()).hexdigest()

# --- APP FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center;'>🛰️ ForensiX Investigation Suite</h1>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1, 1.5, 1])
    
    with col_auth:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # --- MODE: LOGIN ---
        if st.session_state["auth_mode"] == "login":
            st.markdown("### AGENT LOGIN")
            with st.form("login_form"):
                u = st.text_input("AGENT ID")
                p = st.text_input("ACCESS KEY", type="password")
                if st.form_submit_button("AUTHORIZE", use_container_width=True):
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u.lower().strip(), hash_val(p)))
                    if c.fetchone():
                        st.session_state["logged_in"], st.session_state["user"] = True, u.strip()
                        st.rerun()
                    else: st.error("INVALID CREDENTIALS")
                    conn.close()
            
            c1, c2 = st.columns(2)
            if c1.button("Create Account"): st.session_state["auth_mode"] = "register"; st.rerun()
            if c2.button("Forgot Key?"): st.session_state["auth_mode"] = "forgot"; st.rerun()

        # --- MODE: REGISTER ---
        elif st.session_state["auth_mode"] == "register":
            st.markdown("### NEW AGENT ENROLLMENT")
            with st.form("reg_form"):
                new_u = st.text_input("NEW AGENT ID")
                new_p = st.text_input("SET ACCESS KEY", type="password")
                new_r = st.text_input("RECOVERY WORD (e.g. city name)")
                if st.form_submit_button("REGISTER"):
                    try:
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO users VALUES (?, ?, ?)", (new_u.lower().strip(), hash_val(new_p), hash_val(new_r)))
                        conn.commit(); conn.close()
                        st.success("Registration Successful!"); time.sleep(1)
                        st.session_state["auth_mode"] = "login"; st.rerun()
                    except: st.error("ID Already Exists")
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()

        # --- MODE: FORGOT PASSWORD ---
        elif st.session_state["auth_mode"] == "forgot":
            st.markdown("### KEY RECOVERY")
            with st.form("reset_form"):
                u_reset = st.text_input("AGENT ID")
                r_reset = st.text_input("RECOVERY WORD")
                new_p1 = st.text_input("NEW ACCESS KEY", type="password")
                if st.form_submit_button("RESET KEY"):
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE username=? AND recovery=?", (u_reset.lower().strip(), hash_val(r_reset)))
                    if c.fetchone():
                        c.execute("UPDATE users SET password=? WHERE username=?", (hash_val(new_p1), u_reset.lower().strip()))
                        conn.commit(); conn.close()
                        st.success("Key Updated!"); time.sleep(1)
                        st.session_state["auth_mode"] = "login"; st.rerun()
                    else: st.error("Recovery Verification Failed")
            if st.button("Back to Login"): st.session_state["auth_mode"] = "login"; st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- MAIN INVESTIGATION SUITE ---
    col_t, col_c = st.columns([2, 1])
    with col_t: st.markdown('<h2 style="color:#00f2ff;">🛰️ ForensiX Suite</h2>', unsafe_allow_html=True)
    with col_c: clock_spot = st.empty()

    with st.sidebar:
        st.markdown(f"**⚡ {st.session_state['user'].upper()} | NAGPUR**")
        if st.button("🔴 EXIT"): st.session_state["logged_in"] = False; st.rerun()

    # Evidence Upload and Analysis (Insert your existing logic here)
    st.file_uploader("Upload Evidence", type=['jpg', 'png'])

    # --- THE LIVE CLOCK FRAGMENT ---
    @st.fragment(run_every="1s")
    def sync_clock():
        now = datetime.now(IST)
        clock_spot.markdown(f"""
            <div style="text-align: right; border-right: 4px solid #00f2ff; padding-right: 15px;">
                <div style="font-size: 26px; font-weight: bold;">{now.strftime('%H:%M:%S')}</div>
                <div style="font-size: 12px; color: #888;">{now.strftime('%d %b %Y').upper()}</div>
            </div>
        """, unsafe_allow_html=True)
    sync_clock()