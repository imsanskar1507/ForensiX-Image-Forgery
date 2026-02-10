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

# LOCK TIME TO INDIA STANDARD TIME (IST)
IST = pytz.timezone('Asia/Kolkata')

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = "Unknown"
if "case_log" not in st.session_state:
    st.session_state["case_log"] = []

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

init_db()

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0b0d; color: #00f2ff; font-family: 'Courier New', monospace; }
    section[data-testid="stSidebar"] { background-color: #0f1116 !important; border-right: 1px solid #00f2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN FLOW ---
if not st.session_state["logged_in"]:
    st.markdown("<br><h1 style='text-align:center;'>🛰️ ForensiX Login</h1>", unsafe_allow_html=True)
    # (... existing login logic here ...)
    st.session_state["logged_in"] = True # For demonstration
else:
    # --- TOP NAVBAR WITH THE CLASSIC NEON CLOCK ---
    col_title, col_clock = st.columns([2, 1])
    with col_title:
        st.markdown('<h2 style="margin-top:20px; color:#00f2ff; text-shadow: 0 0 5px #00f2ff;">🛰️ ForensiX Investigation Suite</h2>', unsafe_allow_html=True)
    
    with col_clock:
        # Placeholder for the high-refresh neon clock
        clock_placeholder = st.empty()

    # --- SIDEBAR STATUS ---
    with st.sidebar:
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid #00f2ff; margin-bottom: 25px;">
                <h4 style="margin:0; font-size: 14px; opacity: 0.8;">OPERATIVE STATUS</h4>
                <h2 style="margin:0; color: #00f2ff; font-size: 22px;">⚡ {st.session_state['user'].upper()}</h2>
                <p style="margin:10px 0 0 0; font-size: 14px; color: #00f2ff; font-weight: bold;">📍 NAGPUR_MS_IN</p>
            </div>
        """, unsafe_allow_html=True)

    # --- MAIN SCANNER LOGIC ---
    # (... upload and processing code here ...)

    # --- THE LIVE NEON CLOCK REFRESH LOOP ---
    while st.session_state["logged_in"]:
        now_ist = datetime.now(IST)
        
        # This CSS mimics the exact aesthetic of your uploaded image
        clock_placeholder.markdown(f"""
            <div style="text-align: right; padding: 5px;">
                <div style="color: #ffffff; font-size: 11px; letter-spacing: 2px; margin-bottom: 3px; opacity: 0.8;">NAGPUR_MS_IN</div>
                <div style="
                    display: inline-block;
                    padding: 15px 30px; 
                    background: rgba(10, 11, 13, 0.9); 
                    border: 3px solid #00f2ff; 
                    border-radius: 20px; 
                    box-shadow: 0 0 20px rgba(0, 242, 255, 0.6), inset 0 0 15px rgba(0, 242, 255, 0.3);
                ">
                    <div style="color: rgba(0, 242, 255, 0.8); font-size: 10px; letter-spacing: 3px; text-align: center; margin-bottom: 5px;">CLASSIC</div>
                    <div style="
                        font-family: 'Courier New', Courier, monospace; 
                        color: #00f2ff; 
                        font-size: 42px; 
                        font-weight: 900;
                        text-align: center;
                        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
                        line-height: 1;
                    ">
                        {now_ist.strftime('%H:%M:%S')}
                        <span style="font-size: 18px; opacity: 0.8; text-shadow: none;"> {now_ist.strftime('%p')}</span>
                    </div>
                    <div style="
                        color: #00f2ff; 
                        font-size: 16px; 
                        text-align: center; 
                        margin-top: 8px; 
                        letter-spacing: 1px;
                        font-weight: bold;
                    ">
                        {now_ist.strftime('%d %b %Y').upper()}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1)