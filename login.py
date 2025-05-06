# login.py
import streamlit as st
import hashlib

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1f1c2c, #928dab);
        }
        .main {
            background-color: #ffffffdd;
            padding: 2rem;
            border-radius: 12px;
            max-width: 400px;
            margin: 4rem auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        input {
            border-radius: 6px !important;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.image("logo.png", width=120)  # optional: your SaaS logo
    st.title("🔒 Secure Login")
    st.write("Enter your credentials to continue.")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    login_button = st.button("Login")

    st.markdown('</div>', unsafe_allow_html=True)

# Sample user credentials (replace with a DB or secure store in production)
USERS = {
    "admin@example.com": "admin123",  # plain-text password
    "user@example.com": "userpass"
}

# Hash passwords for basic security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    st.title("🔐 Login to Prompt Insights AI")
    st.subheader("Enterprise Data Companion")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if email in USERS and USERS[email] == password:
            st.session_state.logged_in = True
            st.session_state.user = email
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid email or password")

def require_login():
    if not st.session_state.get("logged_in"):
        login()
        st.stop()
