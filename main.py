# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
import openai
import logging
import numpy as np
from datetime import datetime
from supabase import create_client
import re
from styles import load_custom_css
from chart_utils import render_chart, rule_based_parse, generate_insights
from calc_utils import evaluate_calculation, generate_formula_from_prompt, detect_outliers, PREDEFINED_CALCULATIONS, calculate_statistics
from prompt_utils import generate_sample_prompts, generate_prompts_with_llm, prioritize_fields
from utils import classify_columns, load_data, save_dashboard, load_dashboards, save_annotation, load_annotations, delete_dashboard, update_dashboard, load_openai_key, generate_gpt_insight_with_fallback, generate_unique_id, parse_prompt, setup_logging
import streamlit as st
from urllib.parse import urlparse, parse_qs
import hashlib
import time
import json

# Set up logging
logger = setup_logging()

def save_field_types(project_name, field_types):
    """Save field types to a JSON file in the project directory."""
    try:
        field_types_file = f"projects/{project_name}/field_types.json"
        with open(field_types_file, 'w') as f:
            json.dump(field_types, f)
        logger.info("Saved field types for project %s: %s", project_name, field_types)
    except Exception as e:
        logger.error("Failed to save field types for project %s: %s", project_name, str(e))
        raise

def load_field_types(project_name):
    """Load field types from a JSON file in the project directory."""
    try:
        field_types_file = f"projects/{project_name}/field_types.json"
        if os.path.exists(field_types_file):
            with open(field_types_file, 'r') as f:
                field_types = json.load(f)
            logger.info("Loaded field types for project %s: %s", project_name, field_types)
            return field_types
        return None
    except Exception as e:
        logger.error("Failed to load field types for project %s: %s", project_name, str(e))
        return None

# Initialize Supabase (use your project URL and anon key)
supabase = create_client("https://fyyvfaqiohdxhnbdqoxu.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5eXZmYXFpb2hkeGhuYmRxb3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NTA2MTYsImV4cCI6MjA2MzEyNjYxNn0.-h6sm3bgPzxDjxlmPhi5LNzsbhMJiz8-0HX80U7FiZc")

def handle_auth_callback():
    query_params = parse_qs(urlparse(st.experimental_get_query_params().get("url", [""])[0]).query)
    token_hash = query_params.get("token_hash", [None])[0]
    auth_type = query_params.get("type", [None])[0]
    if token_hash and auth_type == "email":
        try:
            response = supabase.auth.verify_otp({"token_hash": token_hash, "type": "email"})
            st.session_state.user_id = response.user.id
            st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
            st.success("Email confirmed!")
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Verification failed: {e}")

if "auth/callback" in st.experimental_get_query_params().get("url", [""])[0]:
    handle_auth_callback()

def login():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            try:
                response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user_id = response.user.id
                st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                # Set session and store tokens
                supabase.auth.set_session(response.session.access_token, response.session.refresh_token)
                st.session_state.access_token = response.session.access_token
                st.session_state.refresh_token = response.session.refresh_token
                # Verify session
                session = supabase.auth.get_session()
                current_user = supabase.auth.get_user()
                logger.info(f"User logged in: {st.session_state.user_id}, Role: {st.session_state.user_role}, Session user: {current_user.user.id if current_user.user else 'None'}, Access token: {session.access_token[:10]}...")
                logger.info(f"Session state after login: {list(st.session_state.keys())}")
                logger.info(f"Stored access_token: {st.session_state.get('access_token', 'None')[:10]}...")
                logger.info(f"Stored refresh_token: {st.session_state.get('refresh_token', 'None')[:10]}...")
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                logger.error(f"Login failed: {str(e)}")
                st.error(f"Login failed: {str(e)}")


def signup():
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            try:
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {"role": "Viewer"}  # Default role
                    }
                })
                st.session_state.user_id = response.user.id
                st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                logger.info(f"User signed up: {st.session_state.user_id}, Role: {st.session_state.user_role}")
                st.success("Signed up! Check your email to confirm.")
                st.rerun()
            except Exception as e:
                logger.error(f"Sign-up failed: {e}")
                st.error(f"Sign-up failed: {e}")

def logout():
    supabase.auth.sign_out()
    st.session_state.user_id = None
    st.session_state.user_role = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    logger.info("User logged out")
    st.success("Logged out!")
    st.rerun()

if "user_id" not in st.session_state:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login()
    with tab2:
        signup()  # Replace with actual signup function or remove if not implemented
else:
    st.write(f"Logged in as: {st.session_state.user_id} ({st.session_state.user_role})")
    if st.button("Logout"):
        logout()

# Load Custom CSS and Override
load_custom_css()
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    .stApp > div > div {
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: white !important;
        width: 320px !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        width: 100% !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #475569 !important;
    }
    [data-testid="stSidebar"] .stSelectbox, .stTextInput, .stExpander, .stInfo {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox>label, .stTextInput>label, .stExpander>label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stExpander div[role="button"] p {
        color: white !important;
    }
    [data-testid="stSidebar"] .stInfo {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stSidebar"] .stInfo div {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div {
        color: black !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div[role="option"] {
        color: black !important;
        background-color: white !important;
    }
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
        text-align: left;
    }
    .styled-table thead tr {
        background-color: #334155;
        color: white;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #334155;
    }
    [data-testid="stSidebar"] .saved-dashboard {
        color: black !important;
    }
    .saved-dashboard {
        color: black !important;
    }
    .sort-button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 0.9em !important;
    }
    .sort-button:hover {
        background-color: #475569 !important;
    }
    .main [data-testid="stExpander"] {
        background-color: #F5F7FA !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        cursor: pointer !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] p {
        color: white !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .main [data-testid="stExpander"] > div[role="button"]:hover {
        background-color: #475569 !important;
    }
    /* New styles for Saved Dashboards tab */
    .dashboard-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .reorder-button {
        background-color: #26A69A !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        border: none !important;
        cursor: pointer !important;
    }
    .reorder-button:hover {
        background-color: #2E7D32 !important;
    }
    .annotation-input {
        background-color: #ECEFF1 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid #B0BEC5 !important;
    }
</style>
""", unsafe_allow_html=True)

# In main.py
def load_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None


# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Session State Init
def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        "chart_history": [],
        "field_types": {},
        "dataset": None,
        "current_project": None,
        "sidebar_collapsed": False,
        "sort_order": {},
        "insights_cache": {},
        "sample_prompts": [],
        "used_sample_prompts": [],
        "sample_prompt_pool": [],
        "last_used_pool_index": 0,
        "onboarding_seen": False,
        "classified": False,
        "last_manual_prompt": None,
        "chart_dimensions": {},
        "chart_cache": {},
        "refresh_dashboards": False,
        "dashboard_order": [],
        "data_loaded": False,
        "loading_progress": 0,
        "last_data_update": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state at startup
initialize_session_state()





# Preprocess Dates (unchanged)
def preprocess_dates(df):
    """Forcefully preprocess date columns with format detection and consistent parsing."""
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%m-%d-%Y', '%d-%m-%Y', '%b %d %Y', '%B %d %Y',
        '%d %b %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'
    ]
    
    for col in df.columns:
        if 'date' in col.lower() or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                # First try to detect the format from a sample of non-null values
                sample = df[col].dropna().head(50)  # Increased sample size for better format detection
                if len(sample) > 0:
                    detected_format = None
                    valid_count = 0
                    best_format = None
                    
                    # Try each format and count valid dates
                    for fmt in date_formats:
                        try:
                            parsed = pd.to_datetime(sample, format=fmt)
                            valid = parsed.notna().sum()
                            if valid > valid_count:
                                valid_count = valid
                                best_format = fmt
                        except:
                            continue
                    
                    if best_format:
                        # Use the best detected format for parsing
                        parsed_col = pd.to_datetime(df[col], format=best_format, errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        logger.info(f"Parsed column '{col}' as datetime using format '{best_format}' with {valid_ratio:.1%} valid dates")
                        
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            logger.info(f"Converted column '{col}' to datetime")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                    else:
                        # If no format detected, try parsing without format
                        parsed_col = pd.to_datetime(df[col], errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            logger.info(f"Converted column '{col}' to datetime using automatic format detection with {valid_ratio:.1%} valid dates")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                else:
                    logger.warning(f"Column '{col}' has no non-null values to detect date format")
            except Exception as e:
                logger.warning(f"Failed to parse date column '{col}': {str(e)}")
    return df

# Sidebar (unchanged)
with st.sidebar:
    st.title("ChartGPT AI")
    
    with st.expander("📂 Projects", expanded=True):
        if "projects" not in st.session_state:
            try:
                st.session_state.projects = os.listdir("projects") if os.path.exists("projects") else []
                logger.info("Loaded existing projects: %s", st.session_state.projects)
            except Exception as e:
                st.error(f"Failed to list projects: {e}")
                logger.error("Failed to list projects: %s", str(e))
                st.session_state.projects = []
        
        projects = st.session_state.projects
        selected_project = st.selectbox("Open Project:", ["(None)"] + projects, key="project_select")
        
        if selected_project != "(None)" and st.session_state.current_project != selected_project:
            try:
                if os.path.exists(f"projects/{selected_project}/dataset.csv"):
                    df = pd.read_csv(f"projects/{selected_project}/dataset.csv")
                    df = preprocess_dates(df)
                    st.session_state.dataset = df
                    st.session_state.current_project = selected_project
                    # Load saved field types
                    saved_field_types = load_field_types(selected_project)
                    if saved_field_types:
                        st.session_state.field_types = saved_field_types
                        st.session_state.classified = True
                    else:
                        st.session_state.classified = False
                    st.session_state.sample_prompts = []
                    st.session_state.used_sample_prompts = []
                    st.session_state.sample_prompt_pool = []
                    st.session_state.last_used_pool_index = 0
                    st.success(f"Opened: {selected_project}")
                    logger.info("Opened project: %s", selected_project)
                else:
                    st.error(f"No dataset found for project {selected_project}. Please upload a dataset.")
                    logger.warning("No dataset found for project: %s", selected_project)
            except Exception as e:
                st.error(f"Failed to load dataset for project {selected_project}: {e}")
                logger.error("Failed to load dataset for project %s: %s", selected_project, str(e))
        
        new_project = st.text_input("New Project Name:", key="new_project_input")
        if st.button("🚀 Create Project", key="create_project"):
            if new_project.strip() != "":
                if new_project not in projects:
                    try:
                        os.makedirs(f"projects/{new_project}")
                        st.session_state.current_project = new_project
                        st.session_state.chart_history = []
                        st.session_state.dataset = None
                        st.session_state.projects.append(new_project)
                        st.session_state.classified = False
                        st.session_state.sample_prompts = []
                        st.session_state.used_sample_prompts = []
                        st.session_state.sample_prompt_pool = []
                        st.session_state.last_used_pool_index = 0
                        st.session_state.field_types = {}
                        # Save empty field types for new project
                        save_field_types(new_project, st.session_state.field_types)
                        st.success(f"Created: {new_project}")
                        logger.info("Created new project: %s", new_project)
                    except Exception as e:
                        st.error(f"Failed to create project: {e}")
                        logger.error("Failed to create project %s: %s", new_project, str(e))
                else:
                    st.error("Project already exists.")
                    logger.warning("Attempted to create project %s, but it already exists", new_project)
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset.copy()
        
        if not st.session_state.classified:
            try:
                dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                df = preprocess_dates(df)  # force parsing of any detected date columns
                for date_col in dates:
                    if date_col in df.columns:
                        try:
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            logger.info(f"Converted {date_col} to datetime format post-classification.")
                        except Exception as e:
                            logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")

                st.session_state.field_types = {
                    "dimension": dimensions,
                    "measure": measures,
                    "date": dates,
                    "id": ids,
                }
                st.session_state.classified = True
                st.session_state.dataset = df
                logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                            st.session_state.current_project, dimensions, measures, dates, ids)
            except Exception as e:
                st.error(f"Failed to classify columns: %s", str(e))
                logger.error("Failed to classify columns: %s", str(e))
                st.stop()
        
        dimensions = st.session_state.field_types.get("dimension", [])
        measures = st.session_state.field_types.get("measure", [])
        dates = st.session_state.field_types.get("date", [])
        
        if measures and dimensions:
            if not st.session_state.sample_prompt_pool:
                st.session_state.sample_prompt_pool = generate_sample_prompts(dimensions, measures, dates, df, max_prompts=10)
            
            if not st.session_state.sample_prompts:
                st.session_state.sample_prompts = st.session_state.sample_prompt_pool[:5]
        
        st.markdown("### Sample Prompts")
        if not st.session_state.sample_prompts:
            st.info("All sample prompts have been used or no valid prompts can be generated due to missing dimensions/measures.")
        else:
            for idx, prompt in enumerate(st.session_state.sample_prompts):
                prompt_text = prompt.split(". ", 1)[1] if ". " in prompt else prompt
                if st.button(prompt, key=f"sidebar_sample_{idx}"):
                    st.session_state.chart_history.append({"prompt": prompt_text})
                    st.session_state.used_sample_prompts.append(prompt_text)
                    st.session_state.sample_prompts.pop(idx)
                    found_new_prompt = False
                    start_index = st.session_state.last_used_pool_index
                    attempts = 0
                    while attempts < len(st.session_state.sample_prompt_pool):
                        next_index = (start_index + attempts) % len(st.session_state.sample_prompt_pool)
                        next_prompt = st.session_state.sample_prompt_pool[next_index]
                        next_prompt_text = next_prompt.split(". ", 1)[1] if ". " in next_prompt else next_prompt
                        already_used = next_prompt_text in st.session_state.used_sample_prompts
                        already_displayed = any(next_prompt_text == p.split(". ", 1)[1] for p in st.session_state.sample_prompts if ". " in p)
                        logger.info(
                            "Checking prompt at index %d: %s (text: %s), already_used=%s, already_displayed=%s",
                            next_index, next_prompt, next_prompt_text, already_used, already_displayed
                        )
                        if already_used or already_displayed:
                            attempts += 1
                            continue
                        st.session_state.sample_prompts.append(next_prompt)
                        st.session_state.last_used_pool_index = (next_index + 1) % len(st.session_state.sample_prompt_pool)
                        found_new_prompt = True
                        logger.info("Selected new prompt at index %d: %s", next_index, next_prompt)
                        break
                    if not found_new_prompt:
                        logger.info("No new unused prompts found in the pool.")
                        st.session_state.last_used_pool_index = 0
                    st.session_state.sample_prompts = [f"{i+1}. {p.split('. ', 1)[1] if '. ' in p else p}" for i, p in enumerate(st.session_state.sample_prompts)]
                    logger.info(
                        "User selected sidebar sample prompt: %s, replaced with: %s",
                        prompt_text,
                        st.session_state.sample_prompts[-1] if st.session_state.sample_prompts else "None"
                    )
                    st.rerun()
    
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **ChartGPT AI** is an AI-powered business intelligence platform that transforms data into actionable insights using natural language. Ask questions, visualize data, and uncover trends effortlessly.
        """)

# Main Content
st.title("ChartGPT AI: Enterprise Insights")
if st.session_state.current_project:
    st.caption(f"Active Project: **{st.session_state.current_project}**")
else:
    st.warning("No project selected. Please create or open a project in the sidebar.")

# Onboarding Modal (unchanged)
if not st.session_state.onboarding_seen:
    with st.container():
        st.markdown("""
        <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 0;'>
            <h3>Welcome to ChartGPT AI! 🎉</h3>
            <p>Transform your data into insights with our AI-powered BI platform. Here's how to get started:</p>
            <ul>
                <li>📂 Create or open a project in the sidebar.</li>
                <li>📊 Upload a CSV or connect to a database.</li>
                <li>💬 Ask questions like "Top 5 Cities by Sales" in the prompt box.</li>
                <li>📈 Explore charts and AI-generated insights.</li>
            </ul>
            <p>Ready to dive in?</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Got it! Let's start.", key="onboarding_close"):
            st.session_state.onboarding_seen = True
            logger.info("User completed onboarding")

# Save Dataset Changes (unchanged)
def save_dataset_changes():
    if st.session_state.current_project and st.session_state.dataset is not None:
        try:
            st.session_state.dataset.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
            # Also save field types
            save_field_types(st.session_state.current_project, st.session_state.field_types)
            logger.info("Saved dataset and field types for project: %s", st.session_state.current_project)
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            logger.error("Failed to save dataset for project %s: %s", st.session_state.current_project, str(e))

# Generate GPT Insights (unchanged)
def generate_gpt_insights(stats, metric, prompt, chart_data, dimension=None, second_metric=None):
    """Generate insights using GPT-3.5-turbo."""
    if not USE_OPENAI:
        return []

    try:
        # Prepare the data summary
        data_summary = {
            "metric": metric,
            "dimension": dimension,
            "second_metric": second_metric,
            "stats": stats,
            "prompt": prompt,
            "data_points": len(chart_data)
        }

        # Create the prompt for GPT
        gpt_prompt = (
            f"Analyze this data visualization and provide 3 concise, insightful observations:\n"
            f"Metric: {data_summary['metric']}\n"
            f"Dimension: {data_summary['dimension']}\n"
            f"Statistics: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, "
            f"Min={stats['min']:.2f}, Max={stats['max']:.2f}\n"
            f"Number of data points: {data_summary['data_points']}\n"
            f"Original prompt: {prompt}\n"
            f"Provide 3 specific, data-driven insights that would be valuable for business users."
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise, actionable insights from data visualizations."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract insights from the response
        insights = [line.strip('- ').strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        return insights[:3]  # Return top 3 insights

    except Exception as e:
        logger.error(f"Failed to generate GPT insights: {str(e)}")
        return []

# In main.py
def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    """
    Generate an executive summary using OpenAI API with a fallback to rule-based summary.
    Args:
        chart_history (list): List of chart objects with prompts and data
        df (pd.DataFrame): The dataset
        dimensions (list): List of dimension columns
        measures (list): List of measure columns
        dates (list): List of date columns
    Returns:
        list: List of summary points
    """
    try:
        logger.info("Generating executive summary for %d charts", len(chart_history))

        # Check if OpenAI is available
        if USE_OPENAI and openai.api_key:
            try:
                # Prepare data summary
                data_summary = {
                    "total_rows": len(df),
                    "dimensions": dimensions,
                    "measures": measures,
                    "dates": dates,
                    "charts": [{"prompt": chart["prompt"], "type": chart.get("chart_type", "Unknown")} for chart in chart_history]
                }

                # Create prompt for OpenAI
                gpt_prompt = (
                    f"Generate a concise executive summary for a data analysis:\n"
                    f"- Dataset size: {data_summary['total_rows']} rows\n"
                    f"- Dimensions: {', '.join(data_summary['dimensions'])}\n"
                    f"- Measures: {', '.join(data_summary['measures'])}\n"
                    f"- Date columns: {', '.join(data_summary['dates']) if data_summary['dates'] else 'None'}\n"
                    f"- Charts analyzed: {len(data_summary['charts'])}\n"
                    f"Chart details:\n" + "\n".join([f"- {chart['prompt']} ({chart['type']})" for chart in data_summary['charts']]) + "\n"
                    f"Provide a 3-paragraph summary highlighting key findings, trends, and actionable recommendations for business strategy."
                )

                # Call OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst creating executive summaries for business stakeholders."},
                        {"role": "user", "content": gpt_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )

                # Extract and split into points
                content = response.choices[0].message.content.strip()
                points = [p.strip() for p in content.split('\n\n') if p.strip()]
                logger.info(f"Generated OpenAI executive summary with {len(points)} points")
                return points
            except Exception as e:
                logger.error(f"OpenAI API call failed for executive summary: {str(e)}")
                # Fall through to fallback mechanism

        # Fallback: Rule-based summary
        logger.info("Using fallback rule-based executive summary")
        if not chart_history:
            return [
                "Analysis Overview: No charts generated yet.",
                "Recommendation: Generate charts in the Dashboard tab to see a summary."
            ]

        summary = []
        key_metrics = {}
        trends = {}
        correlations = {}

        # Aggregate insights from charts
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result is None:
                continue

            chart_data, metric, dimension, working_df, _, _, _ = chart_result
            stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None

            if stats:
                key_metrics[metric] = {
                    'mean': stats['mean'],
                    'max': stats['max'],
                    'min': stats['min'],
                    'std_dev': stats['std_dev']
                }

                if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                    monthly_avg = chart_data.groupby(chart_data[dimension].dt.to_period('M'))[metric].mean()
                    trends[metric] = {
                        'peak': (monthly_avg.idxmax(), monthly_avg.max()),
                        'low': (monthly_avg.idxmin(), monthly_avg.min()),
                        'trend': 'increasing' if monthly_avg.iloc[-1] > monthly_avg.iloc[0] else 'decreasing'
                    }

                for other_metric in measures:
                    if other_metric != metric and other_metric in chart_data.columns:
                        corr = chart_data[[metric, other_metric]].corr().iloc[0, 1]
                        if abs(corr) > 0.5:
                            correlations[(metric, other_metric)] = corr

        # Build summary
        if key_metrics:
            summary.append("**Key Performance Metrics:**")
            for metric, values in key_metrics.items():
                summary.append(f"- {metric}: Average ${values['mean']:.2f} (Range: ${values['min']:.2f} - ${values['max']:.2f})")

        if trends:
            summary.append("\n**Key Trends:**")
            for metric, trend_data in trends.items():
                summary.append(f"- {metric} shows {trend_data['trend']} trend, peaking at ${trend_data['peak'][1]:.2f} in {trend_data['peak'][0]}")

        if correlations:
            summary.append("\n**Key Correlations:**")
            for (metric1, metric2), corr in correlations.items():
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                summary.append(f"- {strength} {direction} correlation ({corr:.2f}) between {metric1} and {metric2}")

        summary.append("\n**Strategic Recommendations:**")
        for metric, trend_data in trends.items():
            if trend_data['trend'] == 'decreasing':
                summary.append(f"- Develop action plan to reverse declining {metric} trend")
            elif trend_data['trend'] == 'increasing':
                summary.append(f"- Scale successful strategies driving {metric} growth")

        for (metric1, metric2), corr in correlations.items():
            if corr > 0.7:
                summary.append(f"- Leverage strong relationship between {metric1} and {metric2} for cross-selling")
            elif corr < -0.7:
                summary.append(f"- Investigate inverse relationship between {metric1} and {metric2}")

        for metric, values in key_metrics.items():
            if values['std_dev'] > values['mean'] * 0.5:
                summary.append(f"- Standardize processes to reduce {metric} variability")
            if values['min'] < values['mean'] * 0.5:
                summary.append(f"- Address underperforming areas in {metric}")

        if not summary:
            summary = ["No significant findings could be summarized from the data."]

        logger.info(f"Generated fallback executive summary with {len(summary)} points")
        return summary

    except Exception as e:
        logger.error("Failed to generate executive summary: %s", str(e))
        return [
            "Analysis Overview: Error generating summary.",
            "Recommendation: Check logs and ensure valid chart data."
        ]

def generate_overall_data_analysis(df, dimensions, measures, dates):
    if not USE_OPENAI:
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

    try:
        stats_summary = []
        for metric in measures:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                stats = calculate_statistics(df, metric)
                stats_summary.append(
                    f"{metric}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}, "
                    f"Q1={stats['q1']:.2f}, median={stats['median']:.2f}, Q3={stats['q3']:.2f}, "
                    f"90th percentile={stats['percentile_90']:.2f}"
                )
        
        top_performers = []
        for dim in dimensions:
            for metric in measures:
                if dim in df.columns and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                    grouped = df.groupby(dim)[metric].mean().sort_values(ascending=False)
                    if not grouped.empty:
                        top = grouped.index[0]
                        top_value = grouped.iloc[0]
                        top_performers.append(f"Top {dim} by {metric}: {top} with average {top_value:.2f}")

        correlations = []
        for i, m1 in enumerate(measures):
            for m2 in measures[i+1:]:
                if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                    corr = df[[m1, m2]].corr().iloc[0, 1]
                    correlations.append(f"Correlation between {m1} and {m2}: {corr:.2f}")

        data_summary = (
            f"Dataset Overview:\n- Dimensions: {', '.join(dimensions)}\n- Measures: {', '.join(measures)}\n- Dates: {', '.join(dates) if dates else 'None'}\n"
            f"Statistics:\n" + "\n".join(stats_summary) + "\n"
            f"Top Performers:\n" + "\n".join(top_performers) + "\n"
            f"Correlations:\n" + "\n".join(correlations)
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing an overall analysis and findings summary for a dataset. Focus on key trends, significant findings, and actionable recommendations."},
                {"role": "user", "content": f"Generate a concise overall data analysis and findings summary (3-5 points) based on the following dataset summary:\n{data_summary}\nHighlight key trends, significant findings, and provide actionable recommendations for business strategy."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        analysis = response.choices[0].message.content.strip().split('\n')
        analysis = [item.strip('- ').strip() for item in analysis if item.strip()]
        logger.info("Generated overall data analysis: %s", analysis)
        return analysis
    except Exception as e:
        logger.error("Failed to generate overall data analysis: %s", str(e))
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

# Display Chart (rewritten)
def display_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """Display a chart with controls and data table."""
    try:
        # Create a unique key for this chart
        chart_key = f"chart_{idx}_{hash(prompt)}"
        
        # Get the chart data and metadata
        try:
            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = render_chart(
                idx, prompt, dimensions, measures, dates, df, sort_order, chart_type
            )
        except ValueError as e:
            st.error(str(e))
            logger.error(f"Chart rendering failed: {str(e)}")
            return
        
        # Initialize chart type in session state if not exists
        if f"chart_type_{chart_key}" not in st.session_state:
            st.session_state[f"chart_type_{chart_key}"] = chart_type or "Bar"
        
        # Create two columns for controls
        col1, col2 = st.columns(2)
        
        # Chart type selection
        with col1:
            selected_chart_type = st.selectbox(
                "Chart Type",
                options=["Bar", "Line", "Scatter", "Map", "Table", "Pie"],
                index=["Bar", "Line", "Scatter", "Map", "Table", "Pie"].index(st.session_state[f"chart_type_{chart_key}"]),
                key=f"chart_type_select_{chart_key}"
            )
            if selected_chart_type != st.session_state[f"chart_type_{chart_key}"]:
                st.session_state[f"chart_type_{chart_key}"] = selected_chart_type
                st.rerun()
        
        # Sort order selection
        with col2:
            sort_order = st.selectbox(
                "Sort Order",
                options=["Ascending", "Descending"],
                index=1 if sort_order == "Descending" else 0,
                key=f"sort_order_{chart_key}"
            )
        
        # Create the chart based on type
        if st.session_state[f"chart_type_{chart_key}"] == "Scatter":
            fig = px.scatter(
                chart_data,
                x=metric,
                y=table_columns[2],  # Use the second metric for y-axis
                color=dimension,  # Color points by dimension
                hover_data=[dimension],  # Add dimension to tooltip
                title=f"{table_columns[2]} vs {metric} by {dimension}",
                labels={metric: metric, table_columns[2]: table_columns[2]},
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=12))
            fig.update_layout(
                xaxis_title=metric,
                yaxis_title=table_columns[2],
                showlegend=True  # Ensure legend is shown
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_scatter")
        elif st.session_state[f"chart_type_{chart_key}"] == "Bar":
            color_col = "Outlier" if "Outlier" in chart_data.columns else None
            fig = px.bar(
                chart_data,
                x=dimension,
                y=metric,
                color=color_col,
                title=f"{metric} by {dimension}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_bar")
        elif st.session_state[f"chart_type_{chart_key}"] == "Line":
            # Get time aggregation for title
            time_agg = kwargs.get("time_aggregation", "month")
            title = f"{metric} by {time_agg.capitalize()}"
            if secondary_dimension:
                title += f" and {secondary_dimension}"
            
            fig = px.line(
                chart_data,
                x=dimension,
                y=metric,
                color=secondary_dimension if secondary_dimension else None,  # Color lines by secondary dimension if present
                title=title,
                template="plotly_white"
            )
            # Add date formatting for date-based line charts
            if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                fig.update_xaxes(
                    tickformat="%b-%Y",
                    tickangle=45,
                    nticks=10
                )
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_line")
        elif st.session_state[f"chart_type_{chart_key}"] == "Map":
            fig = px.choropleth(
                chart_data,
                locations=dimension,
                locationmode="country names",
                color=metric,
                title=f"{metric} by {dimension}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_map")
        elif st.session_state[f"chart_type_{chart_key}"] == "Pie":
            fig = px.pie(
                chart_data,
                names=dimension,
                values=metric,
                title=f"{metric} by {dimension}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_pie")
        else:  # Table view
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table")
        
        # Display insights right below the chart
        try:
            insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
            st.markdown("### Insights")
            for insight in insights:
                st.markdown(f"🔹 {insight}")
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            st.markdown("Unable to generate insights at this time.")
        
        # Display data table in a collapsed expander
        with st.expander("View Data", expanded=False):
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table_data")
            
            if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                st.markdown("### Basic Statistics")
                stats = calculate_statistics(working_df, metric)
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['median']:.2f}")
                    with col3:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{stats['max']:.2f}")
        
    except Exception as e:
        logger.error(f"Error displaying chart: {str(e)}")
        st.error(f"Error displaying chart: {str(e)}")

# Parse IF Statement (unchanged)
def parse_if_statement(formula_str):
    formula_str = formula_str.replace("IF ", "").replace(" THEN ", " ? ").replace(" ELSE ", " : ").replace(" END", "")
    formula_str = formula_str.replace(" AND ", " and ").replace(" OR ", " or ")
    
    def convert_to_ternary(expr):
        if " ? " not in expr or " : " not in expr:
            return expr
        
        parts = expr.split(" ? ", 1)
        if len(parts) < 2:
            logger.error("Invalid IF statement syntax: %s", expr)
            raise ValueError("Invalid IF statement syntax")
        
        condition = parts[0].strip()
        rest = parts[1]
        
        depth = 0
        then_end = -1
        for i, char in enumerate(rest):
            if char == "?":
                depth += 1
            elif char == ":" and depth == 0:
                then_end = i
                break
            elif char == ":":
                depth -= 1
        
        if then_end == -1:
            logger.error("Missing ELSE in IF statement: %s", expr)
            raise ValueError("Missing ELSE in IF statement")
        
        then_part = rest[:then_end].strip()
        else_part = rest[then_end + 1:].strip()
        
        if " ? " in then_part:
            then_part = convert_to_ternary(then_part)
        if " ? " in else_part:
            else_part = convert_to_ternary(else_part)
        
        if then_part.isalpha() or (then_part.startswith('"') and then_part.endswith('"')):
            then_part = f"'{then_part.strip('\"')}'"
        if else_part.isalpha() or (else_part.startswith('"') and else_part.endswith('"')):
            else_part = f"'{else_part.strip('\"')}'"
        
        return f"({then_part} if {condition} else {else_part})"

    try:
        result = convert_to_ternary(formula_str)
        logger.info("Parsed IF statement: %s -> %s", formula_str, result)
        return result
    except Exception as e:
        logger.error("Failed to parse IF statement '%s': %s", formula_str, str(e))
        raise


def recommended_charts_insights_tab():
    st.subheader("📊 Recommended Charts & Insights")
    
    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset first.")
        return
    
    dimensions = st.session_state.field_types.get("dimension", [])
    measures = st.session_state.field_types.get("measure", [])
    dates = st.session_state.field_types.get("date", [])
    
    recommendations = []
    
    # Recommendation 1: Time series for date fields
    if dates and measures:
        date_col = dates[0]
        measure_col = measures[0]
        prompt = f"{measure_col} by {date_col}"
        chart_type = "Line"
        recommendations.append((prompt, chart_type))
    
    # Recommendation 2: Top N analysis for dimensions
    if dimensions and measures:
        dim_col = dimensions[0]
        measure_col = measures[0]
        prompt = f"Top 5 {dim_col} by {measure_col}"
        chart_type = "Bar"
        recommendations.append((prompt, chart_type))
    
    # Recommendation 3: Scatter plot for two measures with a dimension
    if len(measures) >= 2 and dimensions:
        measure1 = measures[0]
        measure2 = measures[1]
        dim_col = dimensions[0]
        prompt = f"{measure1} vs {measure2} by {dim_col}"
        chart_type = "Scatter"
        recommendations.append((prompt, chart_type))
    
    # Recommendation 4: Map for geographical data
    if "Country" in dimensions and measures:
        measure_col = measures[0]
        prompt = f"{measure_col} by Country"
        chart_type = "Map"
        recommendations.append((prompt, chart_type))
    
    if not recommendations:
        st.info("No chart recommendations available based on the dataset structure.")
        return
    
    st.markdown("### Suggested Charts")
    for idx, (prompt, chart_type) in enumerate(recommendations):
        st.markdown(f"**Recommendation {idx + 1}: {prompt} ({chart_type})**")
        try:
            chart_type_key = f"recommended_chart_type_{idx}_{hash(prompt)}"
            
            if chart_type_key not in st.session_state:
                st.session_state[chart_type_key] = chart_type
            
            selected_chart_type = st.selectbox(
                "Chart Type:",
                options=["Bar", "Line", "Scatter", "Map", "Table", "Pie"],
                index=["Bar", "Line", "Scatter", "Map", "Table", "Pie"].index(st.session_state[chart_type_key]),
                key=chart_type_key,
                label_visibility="collapsed"
            )
            
            if selected_chart_type != st.session_state[chart_type_key]:
                st.session_state[chart_type_key] = selected_chart_type
                st.rerun()
            
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df, chart_type=selected_chart_type)
            if chart_result is None:
                st.error(f"Error processing recommendation: {prompt}")
                continue
                
            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
            
            if selected_chart_type == "Line" and pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                chart_data = chart_data.sort_values(by=dimension)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if selected_chart_type == "Bar":
                    fig = px.bar(chart_data, x=dimension, y=metric, labels={dimension: dimension, metric: metric})
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_chart_type == "Line":
                    # Get time aggregation for title
                    time_agg = kwargs.get("time_aggregation", "month")
                    title = f"{metric} by {time_agg.capitalize()}"
                    if secondary_dimension:
                        title += f" and {secondary_dimension}"
                    
                    fig = px.line(chart_data, x=dimension, y=metric, title=title, labels={dimension: dimension, metric: metric})
                    if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                        fig.update_xaxes(
                            tickformat="%b-%Y",
                            tickangle=45,
                            nticks=10
                        )
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_chart_type == "Scatter":
                    if len(table_columns) >= 3:  # We have two metrics
                        fig = px.scatter(
                            chart_data,
                            x=metric,
                            y=table_columns[2],
                            color=dimension,
                            hover_data=[dimension],
                            title=f"{table_columns[2]} vs {metric} by {dimension}",
                            labels={metric: metric, table_columns[2]: table_columns[2]},
                            template="plotly_white"
                        )
                        fig.update_traces(marker=dict(size=12))
                        fig.update_layout(
                            xaxis_title=metric,
                            yaxis_title=table_columns[2],
                            showlegend=True
                        )
                    else:
                        fig = px.scatter(chart_data, x=dimension, y=metric, labels={dimension: dimension, metric: metric})
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_chart_type == "Map":
                    fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric, hover_data=[metric])
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_chart_type == "Table":
                    st.dataframe(chart_data, use_container_width=True)
                elif selected_chart_type == "Pie":
                    fig = px.pie(chart_data, names=dimension, values=metric, labels={dimension: dimension, metric: metric})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Generate and display insights
                try:
                    insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                    st.markdown("### Insights")
                    for insight in insights:
                        st.markdown(f"🔹 {insight}")
                except Exception as e:
                    logger.error(f"Error generating insights: {str(e)}")
                    st.markdown("Unable to generate insights at this time.")
            
            with st.expander("📋 View Chart Data", expanded=False):
                st.markdown("### Data Table")
                st.dataframe(chart_data, use_container_width=True)
                
                if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                    st.markdown("### Basic Statistics")
                    stats = calculate_statistics(working_df, metric)
                    if stats:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{stats['mean']:.2f}")
                        with col2:
                            st.metric("Median", f"{stats['median']:.2f}")
                        with col3:
                            st.metric("Min", f"{stats['min']:.2f}")
                        with col4:
                            st.metric("Max", f"{stats['max']:.2f}")
            
            st.markdown("---")
            
        except Exception as e:
            logger.error(f"Error processing recommendation {idx + 1}: {str(e)}")
            st.error(f"Error processing recommendation: {str(e)}")




def executive_summary_tab(df):
    """
    Render the Executive Summary tab with summary and overall data analysis.
    Args:
        df (pd.DataFrame): The dataset to use for generating the executive summary.
    """
    try:
        logger.info("Entering executive_summary_tab with df shape: %s", str(df.shape) if df is not None else "None")
        st.subheader("📜 Executive Summary")

        if df is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data Manager' tab to view the executive summary.")
            logger.warning("No dataset loaded for executive summary tab")
            return

        # Retrieve dimensions, measures, and dates from session state
        dimensions = st.session_state.field_types.get("dimension", [])
        measures = st.session_state.field_types.get("measure", [])
        dates = st.session_state.field_types.get("date", [])
        logger.info("Dimensions: %s, Measures: %s, Dates: %s", dimensions, measures, dates)

        # Export to PDF button for Executive Summary
        if st.button("📄 Export Executive Summary to PDF", key="export_summary_pdf"):
            try:
                logger.info("Generating LaTeX content for executive summary PDF export")
                latex_content = []
                latex_content.append(r"\documentclass{article}")
                latex_content.append(r"\usepackage{geometry}")
                latex_content.append(r"\usepackage{graphicx}")
                latex_content.append(r"\usepackage{caption}")
                latex_content.append(r"\usepackage{booktabs}")
                latex_content.append(r"\geometry{a4paper, margin=1in}")
                latex_content.append(r"\begin{document}")
                latex_content.append(r"\title{Executive Summary Report}")
                latex_content.append(r"\author{ChartGPT AI}")
                latex_content.append(r"\date{\today}")
                latex_content.append(r"\maketitle")
                latex_content.append(r"\section{Summary of Dashboard Analysis}")

                if not st.session_state.chart_history:
                    latex_content.append("No analysis performed yet. Please generate charts in the 'Dashboard' tab to see a summary.")
                else:
                    summary = generate_executive_summary(st.session_state.chart_history, df, dimensions, measures, dates)
                    latex_content.append(r"\subsection{Key Findings and Recommendations}")
                    latex_content.append(r"\begin{itemize}")
                    for point in summary:
                        latex_content.append(r"\item " + point.replace("_", r"\_"))
                    latex_content.append(r"\end{itemize}")

                latex_content.append(r"\section{Overall Data Analysis and Findings}")
                overall_analysis = generate_overall_data_analysis(df, dimensions, measures, dates)
                latex_content.append(r"\subsection{Overall Insights}")
                latex_content.append(r"\begin{itemize}")
                for point in overall_analysis:
                    latex_content.append(r"\item " + point.replace("_", r"\_"))
                latex_content.append(r"\end{itemize}")

                latex_content.append(r"\end{document}")
                latex_content = "\n".join(latex_content)

                # Export LaTeX content to PDF (handled by system)
                st.session_state["summary_pdf_content"] = latex_content
                st.markdown(
                    f"""
                    <xaiDownloadable
                        filename="Executive_Summary_Report.pdf"
                        contentType="text/latex"
                        content="{latex_content}"
                    />
                    """,
                    unsafe_allow_html=True
                )
                logger.info("Successfully generated LaTeX content for executive summary PDF export")
            except Exception as e:
                logger.error("Failed to generate executive summary PDF: %s", str(e))
                st.error(f"Failed to generate executive summary PDF: {str(e)}")

        # Display the Executive Summary content
        st.markdown("### 📝 Summary of Dashboard Analysis")
        if not st.session_state.chart_history:
            st.markdown("No analysis performed yet. Please generate charts in the 'Dashboard' tab to see a summary.")
            logger.info("No chart history available for executive summary")
        else:
            try:
                summary = generate_executive_summary(st.session_state.chart_history, df, dimensions, measures, dates)
                logger.info("Generated executive summary with %d points", len(summary))
                st.markdown("#### Key Findings and Recommendations")
                for point in summary:
                    st.markdown(f"- {point}")
            except Exception as e:
                logger.error("Error generating executive summary: %s", str(e))
                st.markdown(f"**Executive Summary:**")
                st.markdown(f"Error generating summary: {str(e)}")

        st.markdown("---")
        st.markdown("### 🔍 Overall Data Analysis and Findings")
        show_overall_analysis = st.toggle("Show Overall Data Analysis", value=False)
        if show_overall_analysis:
            try:
                overall_analysis = generate_overall_data_analysis(df, dimensions, measures, dates)
                logger.info("Generated overall data analysis with %d points", len(overall_analysis))
                st.markdown("#### Overall Insights")
                for point in overall_analysis:
                    st.markdown(f"- {point}")
            except Exception as e:
                logger.error("Error generating overall data analysis: %s", str(e))
                st.markdown("**Overall Data Analysis:**")
                st.markdown(f"Error generating analysis: {str(e)}")

        logger.info("Exiting executive_summary_tab")
    except Exception as e:
        logger.error("Unexpected error in executive_summary_tab: %s", str(e))
        st.error(f"Unexpected error in executive summary tab: {str(e)}")

def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    """
    Generate a concise executive summary based on chart history and data analysis.
    Args:
        chart_history (list): List of chart objects with prompts and data
        df (pd.DataFrame): The dataset
        dimensions (list): List of dimension columns
        measures (list): List of measure columns
        dates (list): List of date columns
    Returns:
        list: List of summary points
    """
    try:
        if not chart_history:
            return [
                "Analysis Overview: No charts generated yet.",
                "Recommendation: Generate charts in the Dashboard tab to see a summary."
            ]

        # Collect insights from all charts
        all_insights = []
        key_metrics = {}
        trends = {}
        correlations = {}
        
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result is None:
                continue
                
            chart_data, metric, dimension, working_df, _, _, _ = chart_result
            stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None
            
            if stats:
                # Track key metrics
                if metric not in key_metrics:
                    key_metrics[metric] = {
                        'mean': stats['mean'],
                        'max': stats['max'],
                        'min': stats['min'],
                        'std_dev': stats['std_dev']
                    }
                
                # Track trends for date dimensions
                if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                    monthly_avg = chart_data.groupby(chart_data[dimension].dt.to_period('M'))[metric].mean()
                    trends[metric] = {
                        'peak': (monthly_avg.idxmax(), monthly_avg.max()),
                        'low': (monthly_avg.idxmin(), monthly_avg.min()),
                        'trend': 'increasing' if monthly_avg.iloc[-1] > monthly_avg.iloc[0] else 'decreasing'
                    }
                
                # Track correlations
                for other_metric in measures:
                    if other_metric != metric and other_metric in chart_data.columns:
                        corr = chart_data[[metric, other_metric]].corr().iloc[0, 1]
                        if abs(corr) > 0.5:  # Only track strong correlations
                            correlations[(metric, other_metric)] = corr

        # Generate summary points
        summary = []
        
        # Add key metrics summary
        if key_metrics:
            summary.append("**Key Performance Metrics:**")
            for metric, values in key_metrics.items():
                summary.append(f"- {metric}: Average ${values['mean']:.2f} (Range: ${values['min']:.2f} - ${values['max']:.2f})")
        
        # Add trend analysis
        if trends:
            summary.append("\n**Key Trends:**")
            for metric, trend_data in trends.items():
                summary.append(f"- {metric} shows {trend_data['trend']} trend, peaking at ${trend_data['peak'][1]:.2f} in {trend_data['peak'][0]}")
        
        # Add correlation insights
        if correlations:
            summary.append("\n**Key Correlations:**")
            for (metric1, metric2), corr in correlations.items():
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                summary.append(f"- {strength} {direction} correlation ({corr:.2f}) between {metric1} and {metric2}")
        
        # Add actionable recommendations
        summary.append("\n**Strategic Recommendations:**")
        
        # Add recommendations based on trends
        for metric, trend_data in trends.items():
            if trend_data['trend'] == 'decreasing':
                summary.append(f"- Develop action plan to reverse declining {metric} trend")
            elif trend_data['trend'] == 'increasing':
                summary.append(f"- Scale successful strategies driving {metric} growth")
        
        # Add recommendations based on correlations
        for (metric1, metric2), corr in correlations.items():
            if corr > 0.7:
                summary.append(f"- Leverage strong relationship between {metric1} and {metric2} for cross-selling")
            elif corr < -0.7:
                summary.append(f"- Investigate inverse relationship between {metric1} and {metric2}")
        
        # Add recommendations based on key metrics
        for metric, values in key_metrics.items():
            if values['std_dev'] > values['mean'] * 0.5:  # High variability
                summary.append(f"- Standardize processes to reduce {metric} variability")
            if values['min'] < values['mean'] * 0.5:  # Significant underperformance
                summary.append(f"- Address underperforming areas in {metric}")

        return summary
    except Exception as e:
        logger.error("Failed to generate executive summary: %s", str(e))
        return [
            "Analysis Overview: Error generating summary.",
            "Recommendation: Check logs and ensure valid chart data."
        ]

def generate_overall_data_analysis(df, dimensions, measures, dates):
    """
    Generate overall data analysis and findings.
    Args:
        df (pd.DataFrame): The dataset
        dimensions (list): List of dimension columns
        measures (list): List of measure columns
        dates (list): List of date columns
    Returns:
        list: List of analysis points
    """
    try:
        analysis = []
        
        # Basic dataset overview
        analysis.append(f"**Dataset Overview:** {len(df)} records with {len(dimensions)} dimensions and {len(measures)} measures")
        
        # Key metrics analysis
        for measure in measures:
            if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                stats = calculate_statistics(df, measure)
                analysis.append(f"\n**{measure} Analysis:**")
                analysis.append(f"- Average: ${stats['mean']:.2f}")
                analysis.append(f"- Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                analysis.append(f"- Variability: {stats['std_dev']:.2f} ({(stats['std_dev']/stats['mean']*100):.1f}% of mean)")
        
        # Dimension analysis
        for dimension in dimensions:
            if dimension in df.columns:
                unique_values = df[dimension].nunique()
                analysis.append(f"\n**{dimension} Analysis:**")
                analysis.append(f"- {unique_values} unique values")
                if unique_values < 10:  # For categorical dimensions with few values
                    value_counts = df[dimension].value_counts()
                    analysis.append("- Distribution:")
                    for value, count in value_counts.items():
                        analysis.append(f"  * {value}: {count} ({count/len(df)*100:.1f}%)")
        
        # Date analysis
        for date_col in dates:
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                date_range = df[date_col].max() - df[date_col].min()
                analysis.append(f"\n**{date_col} Analysis:**")
                analysis.append(f"- Date Range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")
                analysis.append(f"- Span: {date_range.days} days")
        
        # Correlation analysis
        if len(measures) > 1:
            analysis.append("\n**Measure Correlations:**")
            for i, m1 in enumerate(measures):
                for m2 in measures[i+1:]:
                    if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                        corr = df[[m1, m2]].corr().iloc[0, 1]
                        if abs(corr) > 0.3:  # Only show meaningful correlations
                            analysis.append(f"- {m1} and {m2}: {corr:.2f}")
        
        return analysis
    except Exception as e:
        logger.error("Failed to generate overall data analysis: %s", str(e))
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

# Tabs
if st.session_state.current_project:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Data", "🛠️ Field Editor", "📈 Dashboard", "📜 Executive Summary", "💾 Saved Dashboards", "📊 Recommended Charts & Insights"])
    
    with tab1:
        st.subheader("📊 Data Management")
        
        st.markdown("### 📤 Upload Dataset")
        with st.container():
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="upload_csv_unique")
            if uploaded_file:
                try:
                    df = load_data(uploaded_file)
                    st.session_state.dataset = df
                    st.session_state.classified = False
                    st.session_state.sample_prompts = []
                    st.session_state.used_sample_prompts = []
                    st.session_state.sample_prompt_pool = []
                    st.session_state.last_used_pool_index = 0
                    st.session_state.field_types = {}
                    df.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
                    st.success("✅ Dataset uploaded!")
                    logger.info("Uploaded dataset for project: %s", st.session_state.current_project)
                except Exception as e:
                    st.error(f"Failed to upload dataset: {str(e)}")
                    logger.error("Failed to upload dataset for project %s: %s", st.session_state.current_project, str(e))        
        st.markdown("### 🔍 Explore Data")
        with st.container():
            if st.session_state.dataset is not None:
                df = st.session_state.dataset
                st.markdown("#### 📄 Dataset Preview (Top 100 Rows)")
                st.dataframe(df.head(100), use_container_width=True)
                
                st.markdown("#### 🧭 Unique Values by Dimension")
                dimensions = st.session_state.field_types.get("dimension", [])
                for dim in dimensions:
                    if dim in df.columns:
                        unique_vals = df[dim].dropna().unique()[:10]
                        st.markdown(f"**{dim}**: {', '.join(map(str, unique_vals))}")
                    else:
                        logger.warning("Dimension %s not found in DataFrame columns", dim)
            else:
                st.info("No dataset uploaded or selected. Please upload a CSV or select a project.")
    
    with tab2:
        if st.session_state.dataset is not None:
            st.subheader("🛠️ Field Editor")
            df = st.session_state.dataset

            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                    df = preprocess_dates(df)  # force parsing of any detected date columns
                    st.session_state.field_types = {
                        "dimension": dimensions,
                        "measure": measures,
                        "date": dates,
                        "id": ids,
                    }
                    st.session_state.classified = True
                    st.session_state.dataset = df
                    logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                                st.session_state.current_project, dimensions, measures, dates, ids)
                except Exception as e:
                    st.error(f"Failed to classify columns: %s", str(e))
                    logger.error("Failed to classify columns: %s", str(e))
                    st.stop()

            st.markdown("### 🔧 Manage Fields and Types")
            with st.expander("Manage Fields and Types", expanded=False):
                for col in df.columns:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{col}**")
                    with col2:
                        current_type = "Other"
                        if col in st.session_state.field_types.get("dimension", []):
                            current_type = "Dimension"
                        elif col in st.session_state.field_types.get("measure", []):
                            current_type = "Measure"
                        elif col in st.session_state.field_types.get("date", []):
                            current_type = "Date"
                        elif col in st.session_state.field_types.get("id", []):
                            current_type = "ID"
                        
                        new_type = st.selectbox(
                            f"Type for {col}",
                            ["Dimension", "Measure", "Date", "ID", "Other"],
                            index=["Dimension", "Measure", "Date", "ID", "Other"].index(current_type),
                            key=f"type_select_{col}",
                            label_visibility="collapsed"
                        )
                        
                        if new_type != current_type:
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            if new_type.lower() != "other":
                                if new_type.lower() in st.session_state.field_types:
                                    st.session_state.field_types[new_type.lower()].append(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.success(f"Field {col} type changed to {new_type}!")
                            logger.info("Changed field type for %s to %s", col, new_type)
                    with col3:
                        new_name = st.text_input(
                            "Rename",
                            value=col,
                            key=f"rename_{col}",
                            label_visibility="collapsed",
                            placeholder="New name"
                        )
                        if new_name and new_name != col:
                            if new_name in df.columns:
                                st.error("Field name already exists!")
                                logger.warning("Attempted to rename field %s to %s, but name already exists", col, new_name)
                            else:
                                df.rename(columns={col: new_name}, inplace=True)
                                st.session_state.dataset = df
                                for t in ["dimension", "measure", "date", "id"]:
                                    if col in st.session_state.field_types.get(t, []):
                                        st.session_state.field_types[t].remove(col)
                                        st.session_state.field_types[t].append(new_name)
                                save_dataset_changes()
                                st.session_state.sample_prompts = []
                                st.session_state.used_sample_prompts = []
                                st.session_state.sample_prompt_pool = []
                                st.session_state.last_used_pool_index = 0
                                st.success(f"Field renamed to {new_name}!")
                                logger.info("Renamed field %s to %s", col, new_name)
                    with col4:
                        if st.button("Delete", key=f"delete_btn_{col}"):
                            df.drop(columns=[col], inplace=True)
                            st.session_state.dataset = df
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.success(f"Field {col} deleted!")
                            logger.info("Deleted field: %s", col)

            st.markdown("### ➕ Create Calculated Fields")
            st.markdown("""
            Create a new calculated field by either describing it in plain English, selecting a predefined template, or directly entering a formula. Supported functions: SUM, AVG, COUNT, STDEV, MEDIAN, MIN, MAX, IF-THEN-ELSE-END.
            """)
            
            input_mode = st.radio(
                "Select Input Mode:",
                ["Prompt-based (Plain English)", "Direct Formula Input"],
                key="calc_input_mode"
            )
            
            st.markdown("#### Predefined Calculation Templates")
            template = st.selectbox(
                "Select a Template (Optional):",
                ["None"] + list(PREDEFINED_CALCULATIONS.keys()),
                key="calc_template"
            )
            
            calc_prompt = ""
            formula_input = ""
            
            if template != "None":
                calc_prompt = PREDEFINED_CALCULATIONS[template]["prompt"]
                formula_input = PREDEFINED_CALCULATIONS[template]["formula"]
            
            dimensions = st.session_state.field_types.get("dimension", [])
            group_by = st.selectbox(
                "Group By (Optional, for 'per' aggregations):",
                ["None"] + dimensions,
                key="calc_group_by"
            )
            group_by = None if group_by == "None" else group_by
            
            if input_mode == "Prompt-based (Plain English)":
                st.markdown("#### Describe Your Calculation")
                measures = st.session_state.field_types.get("measure", [])
                dimensions = st.session_state.field_types.get("dimension", [])
                sample_measure1 = measures[0] if measures else "Measure1"
                sample_measure2 = measures[1] if len(measures) > 1 else "Measure2"
                sample_dimension = dimensions[0] if dimensions else "Dimension1"
                
                examples = [
                    f"Mark {sample_measure1} as High if greater than 1000, otherwise Low",
                    f"Calculate the profit margin as {sample_measure1} divided by {sample_measure2}",
                    f"Flag outliers in {sample_measure1} where {sample_measure1} is more than 2 standard deviations above the average",
                    f"Calculate average {sample_measure1} per {sample_dimension} and flag if above overall average",
                    f"If {sample_measure1} is greater than 500 and {sample_measure2} is positive, then High Performer, else if {sample_measure1} is less than 200, then Low Performer, else Medium"
                ]
                
                st.markdown("Examples:")
                for example in examples:
                    st.markdown(f"- {example}")
                
                calc_prompt = st.text_area("Describe Your Calculation in Plain Text:", value=calc_prompt, key="calc_prompt")
            else:
                st.markdown("#### Enter Formula Directly")
                st.markdown("""
                Enter a formula using exact column names (e.g., Sales, not [Sales]). Examples:
                - IF Sales > 1000 THEN 'High' ELSE 'Low' END
                - Profit / Sales
                - IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END
                - IF AVG(Profit) PER Ship Mode > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END
                """)
                formula_input = st.text_area("Enter Formula:", value=formula_input, key="calc_formula_input")
            
            new_field_name = st.text_input("New Field Name:", key="calc_new_field")
            
            if st.button("Create Calculated Field", key="calc_create"):
                if new_field_name in df.columns:
                    st.error("Field name already exists!")
                    logger.warning("Attempted to create field %s, but name already exists", new_field_name)
                elif not new_field_name:
                    st.error("Please provide a new field name!")
                    logger.warning("User attempted to create a calculated field without a name")
                elif (input_mode == "Prompt-based (Plain English)" and not calc_prompt) or (input_mode == "Direct Formula Input" and not formula_input):
                    st.error("Please provide a calculation description or formula!")
                    logger.warning("User attempted to create a calculated field without a description or formula")
                else:
                    with st.spinner("Processing calculation..."):
                        proceed_with_evaluation = True

                        if input_mode == "Prompt-based (Plain English)":
                            formula = generate_formula_from_prompt(
                                calc_prompt,
                                st.session_state.field_types.get("dimension", []),
                                st.session_state.field_types.get("measure", []),
                                df
                            )
                        else:
                            formula = formula_input
                        
                        if not formula:
                            st.error("Could not generate a formula from the prompt.")
                            logger.warning("Failed to generate formula for prompt: %s", calc_prompt)
                            proceed_with_evaluation = False

                        if proceed_with_evaluation:
                            if '=' in formula:
                                parts = formula.split('=', 1)
                                if len(parts) == 2:
                                    formula = parts[1].strip()
                                else:
                                    st.error("Invalid formula format.")
                                    logger.warning("Invalid formula format: %s", formula)
                                    proceed_with_evaluation = False

                            if proceed_with_evaluation:
                                for col in df.columns:
                                    formula = formula.replace(f"[{col}]", col)
                                
                                working_df = df.copy()
                                formula_modified = formula
                                group_averages = {}
                                overall_avg = None
                                group_dim = None
                                
                                per_match = re.search(r'AVG\((\w+)\)\s+PER\s+(\w+(?:\s+\w+)*)', formula_modified, re.IGNORECASE)
                                if per_match:
                                    agg_col = per_match.group(1)
                                    group_dim = per_match.group(2)
                                    if agg_col in working_df.columns and group_dim in working_df.columns:
                                        overall_avg = working_df[agg_col].mean()
                                        group_averages = working_df.groupby(group_dim)[agg_col].mean().to_dict()
                                        formula_modified = formula_modified.replace(f"AVG({agg_col})", str(overall_avg))
                                        formula_modified = re.sub(r'\s+PER\s+\w+(?:\s+\w+)*', '', formula_modified)
                                    else:
                                        st.error("Invalid columns in PER expression.")
                                        logger.error("Invalid columns in PER expression: %s, %s", agg_col, group_dim)
                                        proceed_with_evaluation = False
                                else:
                                    for col in df.columns:
                                        if f"AVG({col})" in formula_modified:
                                            avg_value = working_df[col].mean()
                                            formula_modified = formula_modified.replace(f"AVG({col})", str(avg_value))
                                        if f"STDEV({col})" in formula_modified:
                                            std_value = working_df[col].std()
                                            formula_modified = formula_modified.replace(f"STDEV({col})", str(std_value))
                                
                                if proceed_with_evaluation:
                                    formula_modified = parse_if_statement(formula_modified)
                                    st.markdown(f"**Formula Used:** `{formula}`")
                                    st.markdown(f"**Processed Formula:** `{formula_modified}`")
                                    try:
                                        def evaluate_row(row):
                                            local_vars = row.to_dict()
                                            if group_averages and group_dim in local_vars:
                                                group_value = group_averages.get(local_vars[group_dim], overall_avg)
                                                condition_expr = formula_modified
                                                for col in df.columns:
                                                    condition_expr = condition_expr.replace(col, str(local_vars.get(col, 0)))
                                                condition_expr = condition_expr.replace(str(overall_avg), str(group_value))
                                                return eval(condition_expr, {"__builtins__": None}, {})
                                            else:
                                                return eval(formula_modified, {"__builtins__": None}, local_vars)

                                        result = working_df.apply(evaluate_row, axis=1)
                                        if result is not None:
                                            df[new_field_name] = result
                                            st.session_state.dataset = df
                                            if pd.api.types.is_numeric_dtype(df[new_field_name]):
                                                if "measure" in st.session_state.field_types:
                                                    st.session_state.field_types["measure"].append(new_field_name)
                                            else:
                                                if "dimension" in st.session_state.field_types:
                                                    st.session_state.field_types["dimension"].append(new_field_name)
                                            save_dataset_changes()
                                            st.session_state.sample_prompts = []
                                            st.session_state.used_sample_prompts = []
                                            st.session_state.sample_prompt_pool = []
                                            st.session_state.last_used_pool_index = 0
                                            st.success(f"New field {new_field_name} created!")
                                            logger.info("Created new calculated field %s with formula: %s", new_field_name, formula)
                                        else:
                                            st.error("Failed to evaluate the formula.")
                                            logger.error("Formula evaluation returned None for prompt: %s", calc_prompt)
                                    except Exception as e:
                                        st.error(f"Error evaluating formula: {str(e)}")
                                        logger.error("Failed to evaluate formula: %s", str(e))

    with tab3:
        st.subheader("📊 Dashboard")
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab.")
        else:
            df = st.session_state.dataset.copy()
            df = preprocess_dates(df)
            
            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                    df = preprocess_dates(df)  # force parsing of any detected date columns
                    st.session_state.field_types = {
                        "dimension": dimensions,
                        "measure": measures,
                        "date": dates,
                        "id": ids,
                    }
                    st.session_state.classified = True
                    st.session_state.dataset = df
                    logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                                st.session_state.current_project, dimensions, measures, dates, ids)
                except Exception as e:
                    st.error(f"Failed to classify columns: {str(e)}")
                    logger.error(f"Failed to classify columns: {str(e)}")
                    st.stop()
            
            dimensions = st.session_state.field_types.get("dimension", [])
            measures = st.session_state.field_types.get("measure", [])
            dates = st.session_state.field_types.get("date", [])
            ids = st.session_state.field_types.get("id", [])
            
            st.markdown("### Recent Charts")
            if len(st.session_state.chart_history) > 10:
                st.session_state.chart_history = st.session_state.chart_history[-5:]
            
            if not measures or not dimensions:
                st.warning(
                    f"Dataset does not meet the requirement of having at least one numeric (measure) and one categorical (dimension) column.\n"
                    f"- **Dimensions (categorical)**: {', '.join(dimensions) if dimensions else 'None'}\n"
                    f"- **Measures (numeric)**: {', '.join(measures) if measures else 'None'}\n"
                    f"- **Dates**: {', '.join(dates) if dates else 'None'}\n"
                    f"- **IDs**: {', '.join(ids) if ids else 'None'}\n"
                    "You can still view the raw data below or adjust column types in the 'Field Editor' tab."
                )
                st.markdown("#### Raw Data Preview")
                st.dataframe(df.head(100))
            else:
                if not st.session_state.chart_history:
                    st.markdown("No recent charts to display.")
                else:
                    if st.button("📄 Export Dashboard to PDF", key="export_dashboard_pdf"):
                        latex_content = []
                        latex_content.append(r"\documentclass{article}")
                        latex_content.append(r"\usepackage{geometry}")
                        latex_content.append(r"\usepackage{graphicx}")
                        latex_content.append(r"\usepackage{caption}")
                        latex_content.append(r"\usepackage{booktabs}")
                        latex_content.append(r"\usepackage{longtable}")
                        latex_content.append(r"\geometry{a4paper, margin=1in}")
                        latex_content.append(r"\begin{document}")
                        latex_content.append(r"\title{Dashboard Report}")
                        latex_content.append(r"\author{ChartGPT AI}")
                        latex_content.append(r"\date{\today}")
                        latex_content.append(r"\maketitle")
                        latex_content.append(r"\section{Recent Charts}")

                        for idx, chart_obj in enumerate(st.session_state.chart_history):
                            prompt_text = chart_obj["prompt"]
                            latex_content.append(r"\subsection{" + prompt_text.replace("_", r"\_") + "}")
                            chart_result = render_chart(idx, prompt_text, dimensions, measures, dates, df)
                            if chart_result:
                                chart_data, metric, dimension, working_df, table_columns, chart_type, _ = chart_result
                                latex_content.append(r"\begin{longtable}{|l|r|}")
                                latex_content.append(r"\hline")
                                latex_content.append(r"\textbf{" + dimension.replace("_", r"\_") + "} & \textbf{" + metric.replace("_", r"\_") + r"} \\")
                                latex_content.append(r"\hline")
                                for _, row in chart_data.iterrows():
                                    latex_content.append(f"{row[dimension]} & {row[metric]:.2f} \\\\")
                                latex_content.append(r"\hline")
                                latex_content.append(r"\end{longtable}")

                                stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None
                                second_metric = next((col for col in measures if col in chart_data.columns and col != metric), None)
                                if stats:
                                    insights = generate_gpt_insights(stats, metric, prompt_text, chart_data, dimension, second_metric)
                                    latex_content.append(r"\subsubsection{Insights}")
                                    latex_content.append(r"\begin{itemize}")
                                    for insight in insights:
                                        latex_content.append(r"\item " + insight.replace("_", r"\_"))
                                    latex_content.append(r"\end{itemize}")

                        latex_content.append(r"\end{document}")
                        latex_content = "\n".join(latex_content)

                        st.session_state["dashboard_pdf_content"] = latex_content
                        st.markdown(
                            f"""
                            <xaiDownloadable
                                filename="Dashboard_Report.pdf"
                                contentType="text/latex"
                                content="{latex_content}"
                            />
                            """,
                            unsafe_allow_html=True
                        )

                    for idx, chart_obj in enumerate(st.session_state.chart_history):
                        prompt_text = chart_obj["prompt"]
                        chart_type = chart_obj.get("chart_type", "")
                        # Only call display_chart once, which will handle both rendering and data display
                        display_chart(idx, prompt_text, dimensions, measures, dates, df)
                        logger.info("Rendered chart %d with prompt: %s", idx, prompt_text)
            
                with st.container():
                    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
                    user_prompt = st.text_input(
                        "💬 Ask about your data (e.g., 'Top 5 Cities by Sales' or 'Find outliers in Sales'):",
                        key="manual_prompt"
                    )
                    st.markdown("#### 🧭 Available Fields")
                    st.markdown(f"**Dimensions:** {', '.join(dimensions)}")
                    st.markdown(f"**Measures:** {', '.join(measures)}")
                    st.markdown(f"**Dates:** {', '.join(dates) if dates else 'None'}")
                    
                    if st.button("🔎 Generate Chart", key="manual_prompt_button"):
                        if user_prompt:
                            # Check for duplicate prompt
                            existing_prompts = [chart["prompt"] for chart in st.session_state.chart_history]
                            if user_prompt not in existing_prompts:
                                st.session_state.last_manual_prompt = user_prompt
                                st.session_state.chart_history.append({"prompt": user_prompt})
                                logger.info("User generated chart with manual prompt: %s", user_prompt)
                                st.rerun()
                            else:
                                st.warning(f"Chart with prompt '{user_prompt}' already exists in history.")
                                logger.info("Prevented duplicate chart prompt: %s", user_prompt)
                    
                    # Dashboard saving section
                    if "user_id" not in st.session_state or st.session_state.user_id is None:
                        st.error("Please log in to save dashboards.")
                    else:
                        dashboard_name = st.text_input(
                            "Dashboard Name",
                            value=f"{st.session_state.current_project}_Dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            key="dashboard_name_input"
                        )
                        if st.button("💾 Save Dashboard", key="save_dashboard"):
                            logger.info("Save Dashboard button clicked")
                            if not st.session_state.chart_history:
                                st.error("No charts to save. Add charts first.")
                            else:
                                try:
                                    # Verify session before saving
                                    session = supabase.auth.get_session()
                                    logger.info(f"Supabase session before save: {session.access_token[:10] + '...' if session else 'None'}")
                                    
                                    # Check for duplicate
                                    existing = load_dashboards(supabase, st.session_state.user_id, st.session_state)
                                    logger.info(f"Checked for duplicates: {len(existing)} existing dashboards")
                                    if not existing.empty:
                                        existing_charts = existing[existing["charts"].apply(lambda x: x == st.session_state.chart_history)]
                                        if not existing_charts.empty:
                                            st.warning(f"Dashboard already exists as '{existing_charts['name'].iloc[0]}'.")
                                            st.session_state.current_dashboard_id = existing_charts["id"].iloc[0]
                                            st.rerun()
                                    dashboard_id = save_dashboard(
                                        supabase,
                                        st.session_state.current_project,
                                        dashboard_name,
                                        st.session_state.chart_history,
                                        st.session_state.user_id,
                                        st.session_state
                                    )
                                    if dashboard_id:
                                        st.success(f"Saved dashboard '{dashboard_name}'")
                                        st.session_state.current_dashboard_id = dashboard_id
                                        logger.info("Saved dashboard '%s' (ID: %s) for project: %s", dashboard_name, dashboard_id, st.session_state.current_project)
                                    else:
                                        st.error("Failed to save dashboard. Please check logs for details.")
                                        logger.error("save_dashboard returned None for project %s", st.session_state.current_project)
                                except Exception as e:
                                    st.error(f"Failed to save dashboard: {str(e)}")
                                    logger.error("Failed to save dashboard for project %s: %s", st.session_state.current_project, str(e))
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        if st.session_state.dataset is not None:
            executive_summary_tab(st.session_state.dataset)
        else:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab to view the executive summary.")

    with tab5:
        st.subheader("💾 All Saved Dashboards")
        
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Upload a dataset in the Data tab.")
        else:
            df = st.session_state.dataset.copy()
            dimensions = st.session_state.field_types.get("dimension", [])
            measures = st.session_state.field_types.get("measure", [])
            dates = st.session_state.field_types.get("date", [])
            
            if "user_id" not in st.session_state or st.session_state.user_id is None:
                st.error("Please log in to view dashboards.")
            else:
                # Load dashboards with cache
                if "dashboards_cache" not in st.session_state or st.session_state.get("refresh_dashboards", False):
                    st.session_state.dashboards_cache = load_dashboards(supabase, st.session_state.user_id, st.session_state)
                    st.session_state.refresh_dashboards = False
                dashboards = st.session_state.dashboards_cache
                logger.info(f"Loaded dashboards: {len(dashboards)} rows")
                
                # Refresh button
                if st.button("🔄 Refresh Dashboards", key="refresh_dashboards_btn"):
                    st.session_state.refresh_dashboards = True
                    st.rerun()
                
                if dashboards.empty:
                    st.markdown("No saved dashboards. Save a dashboard in the Dashboard tab.")
                else:
                    # Search and filters
                    st.markdown("### Filter Dashboards")
                    search_query = st.text_input("Search by name, prompt, or tag", key="dashboard_search")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        project_filter = st.multiselect("Project", options=sorted(dashboards["project_id"].unique()), key="project_filter")
                    with col2:
                        type_filter = st.multiselect("Analytics Type", options=["Sales", "Profit", "Balance", "Other"], key="type_filter")
                    with col3:
                        date_filter = st.date_input("Created After", value=None, key="date_filter")
                    with col4:
                        tag_filter = st.text_input("Tags (comma-separated)", key="tag_filter")
                    
                    # Filter dashboards
                    filtered_dashboards = dashboards
                    if search_query:
                        filtered_dashboards = filtered_dashboards[
                            filtered_dashboards["name"].str.contains(search_query, case=False, na=False) |
                            filtered_dashboards["charts"].apply(lambda charts: any(search_query.lower() in chart.get("prompt", "").lower() for chart in charts))
                        ]
                    if project_filter:
                        filtered_dashboards = filtered_dashboards[filtered_dashboards["project_id"].isin(project_filter)]
                    if type_filter:
                        filtered_dashboards = filtered_dashboards[
                            filtered_dashboards["charts"].apply(lambda charts: any(get_analytics_type(chart) in type_filter for chart in charts))
                        ]
                    if date_filter:
                        filtered_dashboards = filtered_dashboards[
                            pd.to_datetime(filtered_dashboards["created_at"]).dt.date >= date_filter
                        ]
                    if tag_filter:
                        tags = [tag.strip() for tag in tag_filter.split(",")]
                        filtered_dashboards = filtered_dashboards[
                            filtered_dashboards["tags"].apply(lambda t: any(tag in t for tag in tags if t))
                        ]
                    
                    # Tabs for analytics types
                    tabs = st.tabs(["All", "Sales", "Profit", "Balance", "Other"])
                    tab_dashboards = {
                        "All": filtered_dashboards,
                        "Sales": filtered_dashboards[filtered_dashboards["charts"].apply(lambda charts: any(get_analytics_type(chart) == "Sales" for chart in charts))],
                        "Profit": filtered_dashboards[filtered_dashboards["charts"].apply(lambda charts: any(get_analytics_type(chart) == "Profit" for chart in charts))],
                        "Balance": filtered_dashboards[filtered_dashboards["charts"].apply(lambda charts: any(get_analytics_type(chart) == "Balance" for chart in charts))],
                        "Other": filtered_dashboards[filtered_dashboards["charts"].apply(lambda charts: any(get_analytics_type(chart) == "Other" for chart in charts))]
                    }
                    
                    for tab_name, tab_data in tab_dashboards.items():
                        with tabs[["All", "Sales", "Profit", "Balance", "Other"].index(tab_name)]:
                            if tab_data.empty:
                                st.markdown(f"No dashboards found for {tab_name}.")
                                continue
                            
                            # Group by project using containers
                            for project_id in sorted(tab_data["project_id"].unique()):
                                st.markdown(f"### 📂 {project_id} ({len(tab_data[tab_data['project_id'] == project_id])} Dashboards)")
                                with st.container(border=True):
                                    project_dashboards = tab_data[tab_data["project_id"] == project_id]
                                    dashboard_groups = project_dashboards.groupby("id")
                                    if "dashboard_order" not in st.session_state or not st.session_state.dashboard_order:
                                        st.session_state.dashboard_order = list(dashboard_groups.groups.keys())
                                    
                                    for i, dashboard_id in enumerate(st.session_state.dashboard_order):
                                        if dashboard_id not in dashboard_groups.groups:
                                            continue
                                        dashboard_data = dashboard_groups.get_group(dashboard_id)
                                        dashboard_name = dashboard_data["name"].iloc[0]
                                        created_at = dashboard_data["created_at"].iloc[0]
                                        tags = dashboard_data["tags"].iloc[0] if "tags" in dashboard_data.columns else []
                                        
                                        with st.expander(f"Dashboard: {dashboard_name} (Created: {created_at})", expanded=False):
                                            st.markdown(f"**Project**: {project_id}")
                                            st.markdown(f"**Tags**: {', '.join(tags) if tags else 'None'}")
                                            new_tags = st.text_input("Add Tags (comma-separated)", key=f"add_tags_{tab_name}_{dashboard_id}")
                                            if new_tags:
                                                tag_list = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
                                                supabase.table("dashboards").update({"tags": list(set(tags + tag_list))}).eq("id", dashboard_id).execute()
                                                st.session_state.refresh_dashboards = True
                                                st.success(f"Updated tags for {dashboard_name}")
                                                st.rerun()
                                            
                                            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                                            with col1:
                                                if st.button("⬆", key=f"move_up_{tab_name}_{dashboard_id}", disabled=i == 0):
                                                    st.session_state.dashboard_order[i], st.session_state.dashboard_order[i-1] = st.session_state.dashboard_order[i-1], st.session_state.dashboard_order[i]
                                                    st.rerun()
                                            with col2:
                                                if st.button("⬇", key=f"move_down_{tab_name}_{dashboard_id}", disabled=i == len(st.session_state.dashboard_order) - 1):
                                                    st.session_state.dashboard_order[i], st.session_state.dashboard_order[i+1] = st.session_state.dashboard_order[i+1], st.session_state.dashboard_order[i]
                                                    st.rerun()
                                            with col3:
                                                if st.button("🗑️", key=f"delete_dashboard_{tab_name}_{dashboard_id}"):
                                                    supabase.table("dashboards").delete().eq("id", dashboard_id).execute()
                                                    st.session_state.dashboard_order.remove(dashboard_id)
                                                    st.session_state.refresh_dashboards = True
                                                    st.rerun()
                                            with col4:
                                                new_name = st.text_input("Rename", value=dashboard_name, key=f"rename_dashboard_{tab_name}_{dashboard_id}")
                                                if new_name != dashboard_name:
                                                    supabase.table("dashboards").update({"name": new_name}).eq("id", dashboard_id).execute()
                                                    st.session_state.refresh_dashboards = True
                                                    st.success(f"Renamed to {new_name}")
                                                    st.rerun()
                                            
                                            charts = dashboard_data["charts"].iloc[0]
                                            if not charts:
                                                st.warning(f"No charts in dashboard '{dashboard_name}'.")
                                                continue
                                            
                                            st.markdown("### Charts")
                                            for idx, chart in enumerate(charts):
                                                prompt = chart.get("prompt", "")
                                                chart_type = chart.get("chart_type", "")
                                                if not prompt:
                                                    st.warning(f"Chart {idx + 1}: No prompt provided.")
                                                    continue
                                                with st.container():
                                                    st.markdown(f"**Chart {idx + 1}: {prompt} ({chart_type})**")
                                                    new_prompt = st.text_input("Edit Prompt", value=prompt, key=f"edit_prompt_{tab_name}_{dashboard_id}_{idx}")
                                                    if new_prompt != prompt:
                                                        charts[idx]["prompt"] = new_prompt
                                                        supabase.table("dashboards").update({"charts": charts}).eq("id", dashboard_id).execute()
                                                        st.session_state.refresh_dashboards = True
                                                        st.success(f"Prompt updated to '{new_prompt}'")
                                                        st.rerun()
                                                    
                                                    # Render chart and compute insights
                                                    chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
                                                    if chart_result:
                                                        chart_data, metric, dimension, working_df, _, render_type, _ = chart_result
                                                        # Only call display_chart which will handle both rendering and data display
                                                        display_chart(idx, prompt, dimensions, measures, dates, df)
                                                        stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None
                                                        second_metric = next((col for col in measures if col in chart_data.columns and col != metric), None)
                                                        insights = []
                                                        if stats and second_metric:
                                                            insights = generate_gpt_insights(stats, metric, prompt, chart_data, dimension, second_metric)
                                                            logger.info("Generated insights for chart %d: %s", idx, insights)
                                                        else:
                                                            logger.warning("No insights generated for chart %d: stats=%s, second_metric=%s", idx, stats, second_metric)
                                                        show_insights = st.toggle("Show Insights", key=f"show_insights_{tab_name}_{dashboard_id}_{idx}")
                                                        if show_insights:
                                                            with st.container():
                                                                st.markdown("##### Insights")
                                                                if insights:
                                                                    for insight in insights:
                                                                        st.markdown(f"- {insight}")
                                                                else:
                                                                    st.markdown("No insights available for this chart.")
                                                        show_data = st.toggle("Show Data", key=f"show_data_{tab_name}_{dashboard_id}_{idx}")
                                                        if show_data:
                                                            with st.container():
                                                                st.markdown("##### Chart Data")
                                                                if chart_data is not None and not chart_data.empty:
                                                                    st.dataframe(chart_data)
                                                                else:
                                                                    st.markdown("No data available to display.")
                                                        # Add Remove Chart button
                                                        if st.button("🗑️ Remove Chart", key=f"remove_chart_{tab_name}_{dashboard_id}_{idx}"):
                                                            charts.pop(idx)
                                                            supabase.table("dashboards").update({"charts": charts}).eq("id", dashboard_id).execute()
                                                            st.session_state.refresh_dashboards = True
                                                            st.success(f"Removed chart '{prompt}' from dashboard '{dashboard_name}'")
                                                            st.rerun()
                                                        logger.info("Rendered chart %d for dashboard %s: %s", idx, dashboard_id, prompt)
                                                    else:
                                                        st.error(f"Failed to render chart for prompt: '{prompt}'")

    with tab6:
        recommended_charts_insights_tab()

else:
    st.info("Please select a project to continue.")

