# style.py
import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }

    /* Global Layout */
    body {
        background-color: #F5F7FA;
        color: #263238;
    }
    .block-container {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        max-width: 1400px;
        margin: 0 auto;
    }
    .main-content {
        padding-bottom: 120px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #263238;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    h1 {
        font-size: 2.25rem;
    }
    h2 {
        font-size: 1.75rem;
    }
    h3 {
        font-size: 1.25rem;
    }
    p, span, label, div {
        color: #37474F;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #263238;
        color: #ECEFF1;
        padding: 1.5rem;
        width: 280px !important;
        transition: width 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #26A69A;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
        text-align: left;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #2E7D32;
    }
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stTextInput input {
        background-color: #37474F;
        color: #ECEFF1;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ECEFF1;
    }
    .sidebar-collapsed {
        width: 80px !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #26A69A;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2E7D32;
    }

    /* Inputs */
    input, textarea, .stTextInput input, .stSelectbox {
        background-color: #ECEFF1;
        color: #263238;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #B0BEC5;
    }

    /* Fixed Prompt Box */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #FFFFFF;
        padding: 1rem 2rem;
        z-index: 1000;
        border-top: 1px solid #B0BEC5;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
    }
    .fixed-bottom .stTextInput, .fixed-bottom .stButton {
        margin-bottom: 0;
    }

    /* Chart Container */
    .chart-container {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #26A69A;
    }
    .chart-container .stSelectbox {
        width: 150px !important;
        float: right;
    }

    /* Filter Container */
    .filter-container {
        background-color: #F5F7FA;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid #B0BEC5;
    }
    .filter-container p {
        margin: 0;
        font-size: 0.9rem;
        color: #37474F;
    }

    /* Sample Prompts */
    .sample-prompts {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .sample-prompts button {
        background-color: #ECEFF1;
        color: #263238;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        border: 1px solid #B0BEC5;
    }
    .sample-prompts button:hover {
        background-color: #26A69A;
        color: #FFFFFF;
    }

    /* Available Columns */
    .available-columns {
        background-color: #F5F7FA;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #B0BEC5;
    }
    .available-columns p {
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #37474F;
    }
    .available-columns strong {
        color: #26A69A;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEFF1;
        color: #263238;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #26A69A;
        color: #FFFFFF;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
        [data-testid="stSidebar"] {
            width: 200px !important;
        }
        .chart-container .stSelectbox {
            width: 100% !important;
            float: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)