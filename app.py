import streamlit as st
import sqlite3
import hashlib
import os
import time

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Pre-seed with your master account if it doesn't exist
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
    except sqlite3.IntegrityError:
        return False

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username.lower().strip(), hashed_pw))
    result = c.fetchone()
    conn.close()
    return result

# Initialize the database on startup
init_db()

# --- LOGIN / REGISTRATION UI ---
if not st.session_state.get("logged_in"):
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        # Tab system for Login vs Register
        tab1, tab2 = st.tabs(["🔒 AGENT LOGIN", "📝 REGISTER AGENT"])
        
        with tab1:
            user = st.text_input("AGENT ID", key="login_user")
            pw = st.text_input("ACCESS KEY", type="password", key="login_pw")
            if st.button("AUTHORIZE SYSTEM"):
                if check_user(user, pw):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = user
                    st.success("ACCESS GRANTED")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("CREDENTIALS NOT RECOGNIZED")
        
        with tab2:
            new_user = st.text_input("NEW AGENT ID", key="reg_user")
            new_pw = st.text_input("SET ACCESS KEY", type="password", key="reg_pw")
            confirm_pw = st.text_input("CONFIRM KEY", type="password")
            
            if st.button("CREATE CREDENTIALS"):
                if new_pw != confirm_pw:
                    st.error("KEYS DO NOT MATCH")
                elif len(new_pw) < 6:
                    st.error("KEY MUST BE AT LEAST 6 CHARACTERS")
                else:
                    if add_user(new_user, new_pw):
                        st.success("AGENT REGISTERED. YOU CAN NOW LOGIN.")
                    else:
                        st.error("AGENT ID ALREADY EXISTS")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # --- YOUR main_app() CODE STARTS HERE ---
    st.sidebar.success(f"Agent {st.session_state['user'].upper()} Online")
    # ... (rest of your forensic engine code)