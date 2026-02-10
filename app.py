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
st.set_page_config(page_title="ForensiX-Image Forgery Detector", layout="wide", page_icon="🕵️")

# Force Time Zone to IST (India Standard Time)
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
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

# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, recovery TEXT)')
    c.execute("SELECT * FROM users WHERE username='sanskar'")
    if not c.fetchone():
        hp = hashlib.sha256("detective2026".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("sanskar", hp))
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

# --- CSS STYLING (Preserving your original UI) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { 
        background-color: #0a0b0d; 
        color: #ffffff; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* REMOVE FORM BORDERS to keep your UI clean */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    /* Your Original Login Box Styling */
    .login-box {
        background: rgba(15, 17, 22, 0.8) !important; 
        border: 1px solid #00f2ff; 
        border-radius: 15px; 
        padding: 30px;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center; color:#00f2ff;'>🛰️ ForensiX Investigation Suite</h1>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 1.5, 1])
    
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("### AUTHORIZATION REQUIRED")
        
        # FORM WRAPPER: Enables "Enter" key without changing UI look
        with st.form("login_gate", clear_on_submit=False):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            
            # This button will trigger on "Enter"
            submitted = st.form_submit_button("AUTHORIZE SESSION", use_container_width=True)
            
            if submitted:
                if check_user(u_in, p_in):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u_in.strip()
                    log_forensic_action(f"Agent {u_in.upper()} authorized session.")
                    st.rerun()
                else:
                    st.error("INVALID CREDENTIALS")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD UI (The part that was working fine) ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin-top:10px; color:#00f2ff;">🛰️ ForensiX Suite</h2>', unsafe_allow_html=True)
    
    with col_clock:
        clock_placeholder = st.empty()

    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 25px;">
                <h4 style="margin:0; font-size: 14px; opacity: 0.8;">OPERATIVE STATUS</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 22px;">⚡ {st.session_state['user'].upper()}</h2>
                <p style="margin:10px 0 0 0; font-size: 14px; color: #00f2ff; font-weight: bold;">📍 NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🔴 EXIT SYSTEM"):
            st.session_state["logged_in"] = False
            st.rerun()

    # --- IMAGE ANALYSIS ENGINE ---
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        for f in files:
            # Your existing ELA/Heatmap/Histogram code goes here
            st.write(f"Logged: {f.name}")

    # --- LIVE CLOCK LOOP ---
    while st.session_state["logged_in"]:
        now_ist = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; border-right: 4px solid #00f2ff; padding-right: 15px; margin-top: 5px;">
                <div style="color: #00f2ff; font-size: 26px; font-family: 'monospace'; font-weight: bold;">
                    {now_ist.strftime('%H:%M:%S')}
                </div>
                <div style="color: #888888; font-size: 12px; font-weight: bold; letter-spacing: 2px;">
                    {now_ist.strftime('%d %b %Y').upper()} | NAGPUR_MS_IN
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)