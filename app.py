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

# --- DATABASE ENGINE ---
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

# --- REFINED CSS (Background + Enter Key Support) ---
# We apply the background image directly to the .stApp container
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(10, 11, 13, 0.8), rgba(10, 11, 13, 0.9)), 
                    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; 
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Keep the form invisible but functional for Enter key */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }

    .login-box {
        background: rgba(15, 17, 22, 0.85) !important; 
        backdrop-filter: blur(10px);
        border: 2px solid #00f2ff; 
        border-radius: 15px; 
        padding: 35px;
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.3);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #00f2ff !important;
        border: 1px solid #00f2ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br><h1 style='text-align:center; color:#00f2ff; text-shadow: 2px 2px 10px #000;'>🛰️ ForensiX Investigation Suite</h1>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 1.5, 1])
    
    with col_l2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#00f2ff; margin-bottom:20px;'>SECURE ACCESS</h3>", unsafe_allow_html=True)
        
        # This form captures the 'Enter' key press
        with st.form("forensic_login", clear_on_submit=False):
            u_in = st.text_input("AGENT ID")
            p_in = st.text_input("ACCESS KEY", type="password")
            
            # Form submit button styled as your login button
            submitted = st.form_submit_button("AUTHORIZE SESSION", use_container_width=True)
            
            if submitted:
                if check_user(u_in, p_in):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = u_in.strip()
                    log_forensic_action(f"Agent {u_in.upper()} authorized session.")
                    st.rerun()
                else:
                    st.error("ACCESS DENIED: INVALID CREDENTIALS")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- DASHBOARD HEADER ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin-top:10px; color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">🛰️ ForensiX Suite</h2>', unsafe_allow_html=True)
    
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
        if st.button("🔴 TERMINATE SESSION"):
            st.session_state["logged_in"] = False
            st.rerun()

    # --- IMAGE ANALYSIS LOGIC ---
    files = st.file_uploader("UPLOAD EVIDENCE", type=["jpg", "png"], accept_multiple_files=True)
    # (... existing scan and analysis logic ...)

    # --- THE LIVE CLOCK REFRESH LOOP (LOCKED TO IST) ---
    while st.session_state["logged_in"]:
        now_ist = datetime.now(IST)
        clock_placeholder.markdown(f"""
            <div style="text-align: right; border-right: 4px solid #00f2ff; padding-right: 15px; margin-top: 5px;">
                <div style="color: #00f2ff; font-size: 26px; font-family: 'monospace'; font-weight: bold;">
                    {now_ist.strftime('%H:%M:%S')}
                </div>
                <div style="color: #aaaaaa; font-size: 12px; font-weight: bold; letter-spacing: 2px;">
                    {now_ist.strftime('%d %b %Y').upper()} | NAGPUR_MS_IN
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)