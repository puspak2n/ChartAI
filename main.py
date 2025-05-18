# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
import openai
import logging
import numpy as np
import re
from styles import load_custom_css
from chart_utils import render_chart, rule_based_parse
from calc_utils import evaluate_calculation, generate_formula_from_prompt, detect_outliers, PREDEFINED_CALCULATIONS, calculate_statistics
from prompt_utils import generate_sample_prompts, generate_prompts_with_llm, prioritize_fields
from utils import classify_columns, load_data, save_dashboard, load_openai_key

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page Config and CSS
st.set_page_config(page_title="ChartGPT AI", page_icon="📊", layout="wide")
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
    /* Enhanced Sidebar Styling */
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
    /* Fix dropdown text visibility */
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div {
        color: black !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div[role="option"] {
        color: black !important;
        background-color: white !important;
    }
    /* Style for tables */
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
    /* Fix saved dashboard text color */
    [data-testid="stSidebar"] .saved-dashboard {
        color: black !important;
    }
    .saved-dashboard {
        color: black !important;
    }
    /* Style for sort button */
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
    /* Fix for main content expander collapse button */
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
</style>
""", unsafe_allow_html=True)


def load_openai_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except KeyError:
        return None

# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Session State Init
if "chart_history" not in st.session_state:
    st.session_state.chart_history = []
if "field_types" not in st.session_state:
    st.session_state.field_types = {}
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False
if "sort_order" not in st.session_state:
    st.session_state.sort_order = {}
if "insights_cache" not in st.session_state:
    st.session_state.insights_cache = {}
if "sample_prompts" not in st.session_state:
    st.session_state.sample_prompts = []
if "used_sample_prompts" not in st.session_state:
    st.session_state.used_sample_prompts = []
if "sample_prompt_pool" not in st.session_state:
    st.session_state.sample_prompt_pool = []
if "last_used_pool_index" not in st.session_state:
    st.session_state.last_used_pool_index = 0
if "onboarding_seen" not in st.session_state:
    st.session_state.onboarding_seen = False
if "classified" not in st.session_state:
    st.session_state.classified = False
if "last_manual_prompt" not in st.session_state:
    st.session_state.last_manual_prompt = None
if "chart_dimensions" not in st.session_state:
    st.session_state.chart_dimensions = {}

# Generate Sample Prompts (unchanged)
def generate_sample_prompts(dimensions, measures, dates, df, max_prompts=10):
    prompts = []
    sample_segment = df['Segment'].iloc[0] if 'Segment' in df.columns else None
    sample_region = df['Region'].iloc[0] if 'Region' in df.columns else None
    sample_subcategory = df['Sub-Category'].iloc[0] if 'Sub-Category' in df.columns else None
    sample_customer = df['Customer Name'].iloc[0] if 'Customer Name' in df.columns else None
    sample_category = df['Category'].iloc[0] if 'Category' in df.columns else None
    sample_city = df['City'].iloc[0] if 'City' in df.columns else None
    sample_market = df['Market'].iloc[0] if 'Market' in df.columns else None
    
    if dates and measures:
        date_col = dates[0]
        for i, measure in enumerate(measures[:2]):
            dim = dimensions[i % len(dimensions)]
            prompt = f"Show {measure} trend over {date_col} by {dim}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if dimensions and measures:
        for i, dim in enumerate(dimensions[:3]):
            prompt = f"Compare Profit Margin by {dim}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if dimensions and measures:
        dim = 'City' if 'City' in dimensions else dimensions[0]
        for i, measure in enumerate(measures[:2]):
            if sample_customer and 'Customer Name' in dimensions:
                prompt = f"Top 5 {dim} by {measure} for {sample_customer}"
            else:
                prompt = f"Top 5 {dim} by {measure}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if measures and dimensions:
        dim = 'Sub-Category' if 'Sub-Category' in dimensions else dimensions[0]
        for i, measure in enumerate(measures[:2]):
            prompt = f"Find outliers in {measure} by {dim}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if len(measures) >= 2 and dimensions:
        dim = 'Market' if 'Market' in dimensions else dimensions[0]
        for i in range(min(2, len(measures) // 2)):
            measure1 = measures[i]
            measure2 = measures[(i + 1) % len(measures)]
            prompt = f"Compare {measure1} and {measure2} by {dim}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if dimensions and measures:
        dim = 'Region' if 'Region' in dimensions else dimensions[1 % len(dimensions)]
        for i, measure in enumerate(measures[1:3]):
            prompt = f"Top 3 {dim} by {measure}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    if dimensions and measures:
        dim = 'Category' if 'Category' in dimensions else dimensions[2 % len(dimensions)]
        for i, measure in enumerate(measures[2:4]):
            prompt = f"{measure} by {dim}"
            if prompt not in prompts:
                prompts.append(prompt)
    
    prompts = prompts[:max_prompts]
    numbered_prompts = [f"{i+1}. {prompt}" for i, prompt in enumerate(prompts)]
    logger.info("Generated rule-based sample prompts: %s", numbered_prompts)
    return numbered_prompts

# Preprocess Dates (unchanged)
def preprocess_dates(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                non_na_ratio = converted.notna().mean()
                if non_na_ratio > 0.8:
                    df[col] = converted
                    logger.info("Preprocessed %s as potential date column (non-NaN ratio=%.2f)", col, non_na_ratio)
            except Exception as e:
                logger.debug("Could not preprocess %s as date column: %s", col, str(e))
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
                    st.success(f"Opened: {selected_project}")
                    logger.info("Opened project: %s", selected_project)
                    st.session_state.classified = False
                    st.session_state.sample_prompts = []
                    st.session_state.used_sample_prompts = []
                    st.session_state.sample_prompt_pool = []
                    st.session_state.last_used_pool_index = 0
                    st.session_state.field_types = {}
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
        df = preprocess_dates(df)
        st.session_state.dataset = df
        
        if not st.session_state.classified:
            try:
                dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
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
            logger.info("Saved dataset changes for project: %s", st.session_state.current_project)
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            logger.error("Failed to save dataset for project %s: %s", st.session_state.current_project, str(e))

# Generate GPT Insights (unchanged)
def generate_gpt_insights(stats, metric, prompt, chart_data, dimension=None, second_metric=None):
    cache_key = f"{prompt}_{metric}_{dimension}_{chart_data.to_json()}"
    if cache_key in st.session_state.insights_cache:
        logger.info("Retrieved cached insights for prompt: %s", prompt)
        return st.session_state.insights_cache[cache_key]

    if not USE_OPENAI:
        insights = [
            f"The average {metric} is {stats['mean']:.2f}.",
            f"The standard deviation of {metric} is {stats['std_dev']:.2f}, indicating the spread of data.",
            f"The 25th percentile (Q1) of {metric} is {stats['q1']:.2f}, median is {stats['median']:.2f}, and 75th percentile (Q3) is {stats['q3']:.2f}.",
            f"The 90th percentile of {metric} is {stats['percentile_90']:.2f}, showing the top range."
        ]
        st.session_state.insights_cache[cache_key] = insights
        return insights
    
    try:
        data_summary = f"Statistics for {metric}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}, Q1={stats['q1']:.2f}, median={stats['median']:.2f}, Q3={stats['q3']:.2f}, 90th percentile={stats['percentile_90']:.2f}."
        if dimension and metric in chart_data.columns and dimension in chart_data.columns:
            grouped_data = chart_data.groupby(dimension)[metric].mean().sort_values(ascending=False)
            top_performer = grouped_data.index[0] if not grouped_data.empty else "N/A"
            top_value = grouped_data.iloc[0] if not grouped_data.empty else 0
            bottom_performer = grouped_data.index[-1] if not grouped_data.empty else "N/A"
            bottom_value = grouped_data.iloc[-1] if not grouped_data.empty else 0
            data_summary += f" By {dimension}, the top performer is {top_performer} with an average {metric} of {top_value:.2f}, and the bottom performer is {bottom_performer} with an average {metric} of {bottom_value:.2f}."
        if second_metric and second_metric in chart_data.columns and metric in chart_data.columns:
            correlation = chart_data[[metric, second_metric]].corr().iloc[0, 1]
            data_summary += f" The correlation between {metric} and {second_metric} is {correlation:.2f}."

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise, diverse, and actionable insights based on statistical data. Avoid repetition and focus on varied perspectives such as trends, comparisons, outliers, and balanced strategies across categories."},
                {"role": "user", "content": f"Generate 3-5 concise, actionable insights for the prompt '{prompt}'. {data_summary} Provide diverse insights, including comparisons between categories, trends, outlier analysis, and strategies to balance performance across categories. Avoid repetitive suggestions."}
            ],
            max_tokens=150,
            temperature=0.7
        )
        insights = response.choices[0].message.content.strip().split('\n')
        insights = [insight.strip('- ').strip() for insight in insights if insight.strip()]
        logger.info("Successfully generated GPT insights for prompt: %s", prompt)
        st.session_state.insights_cache[cache_key] = insights
        return insights
    except Exception as e:
        logger.error("Failed to generate GPT insights: %s", str(e))
        insights = [
            f"The average {metric} is {stats['mean']:.2f}.",
            f"The standard deviation of {metric} is {stats['std_dev']:.2f}, indicating the spread of data.",
            f"The 25th percentile (Q1) of {metric} is {stats['q1']:.2f}, median is {stats['median']:.2f}, and 75th percentile (Q3) is {stats['q3']:.2f}.",
            f"The 90th percentile of {metric} is {stats['percentile_90']:.2f}, showing the top range."
        ]
        st.session_state.insights_cache[cache_key] = insights
        return insights

# Generate Executive Summary (unchanged)
def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    if not USE_OPENAI:
        return [
            "Analysis Overview: Reviewed multiple aspects of the dataset.",
            "Key Finding: Sales performance varies significantly across categories.",
            "Recommendation: Focus on high-performing categories to maximize revenue."
        ]

    try:
        all_insights = []
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result is None:
                logger.warning("Skipping chart %d with prompt '%s' due to rendering failure", idx, prompt)
                all_insights.append(f"**Prompt: {prompt}**\n- Chart rendering failed. Unable to generate insights.")
                continue
            chart_data, metric, dimension, working_df, table_columns, chart_type, _ = chart_result
            stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None
            second_metric = None
            for col in measures:
                if col in chart_data.columns and col != metric:
                    second_metric = col
                    break
            if stats:
                insights = generate_gpt_insights(stats, metric, prompt, chart_data, dimension, second_metric)
                all_insights.append(f"**Prompt: {prompt}**\n" + "\n".join([f"- {insight}" for insight in insights]))
            else:
                all_insights.append(f"**Prompt: {prompt}**\n- No statistical insights available.")

        insights_summary = "\n\n".join(all_insights) if all_insights else "No insights generated yet."
        
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst tasked with creating an executive summary. Summarize key findings and provide high-level recommendations based on the insights provided."},
                {"role": "user", "content": f"Generate a concise executive summary (3-5 points) based on the following insights:\n{insights_summary}\nFocus on overarching themes, key findings, and actionable recommendations for business strategy."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip().split('\n')
        summary = [item.strip('- ').strip() for item in summary if item.strip()]
        logger.info("Generated executive summary: %s", summary)
        return summary
    except Exception as e:
        logger.error("Failed to generate executive summary: %s", str(e))
        return [
            "Analysis Overview: Reviewed multiple aspects of the dataset.",
            "Key Finding: Sales performance varies significantly across categories.",
            "Recommendation: Focus on high-performing categories to maximize revenue."
        ]

# Generate Overall Data Analysis (unchanged)
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

# Display Chart (unchanged)
def display_chart(idx, prompt, dimensions, measures, dates, df):
    try:
        if idx not in st.session_state.sort_order:
            st.session_state.sort_order[idx] = "Descending"

        col1, col2, col3, col4 = st.columns([2.5, 1, 0.5, 0.5])
        
        with col1:
            st.subheader(prompt)

        with col2:
            chart_types = ["Bar", "Line", "Scatter", "Map", "Table", "Pie"]
            default_index = 0
            if " vs " in prompt.lower():
                default_index = chart_types.index("Scatter")
            elif "by country" in prompt.lower():
                default_index = chart_types.index("Map")
            selected_chart_type = st.selectbox(
                "Chart Type:",
                options=chart_types,
                index=default_index,
                key=f"chart_type_{idx}",
                label_visibility="collapsed"
            )

        with col3:
            sort_label = f"Sort: {st.session_state.sort_order[idx]}"
            if st.button(sort_label, key=f"sort_button_{idx}", help="Toggle sorting order"):
                st.session_state.sort_order[idx] = "Ascending" if st.session_state.sort_order[idx] == "Descending" else "Descending"
                logger.info("Toggled sort order for chart %d to %s", idx, st.session_state.sort_order[idx])
            st.markdown(
                f"""
                <script>
                    document.querySelector('button[kind="secondary"][id="sort_button_{idx}"]').classList.add('sort-button');
                </script>
                """,
                unsafe_allow_html=True
            )

        with col4:
            if st.button("🗑️", key=f"remove_chart_{idx}"):
                st.session_state.chart_history.pop(idx)
                if idx in st.session_state.sort_order:
                    del st.session_state.sort_order[idx]
                logger.info("Removed chart %d with prompt: %s", idx, prompt)
                st.rerun()
                return

        chart_result = render_chart(idx, prompt, dimensions, measures, dates, df, sort_order=st.session_state.sort_order[idx], chart_type=selected_chart_type)
        if chart_result is None:
            st.error(f"Failed to render chart for prompt: '{prompt}'. Please check the prompt or data.")
            logger.error("Chart result is None for prompt: %s", prompt)
            return
        
        chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension = chart_result
        
        second_metric = None
        for col in measures:
            if col in chart_data.columns and col != metric:
                second_metric = col
                break
        
        logger.info("Chart data before processing: rows=%d, columns=%s", len(chart_data), chart_data.columns.tolist())
        
        unique_values = len(chart_data[dimension].unique()) if dimension in chart_data.columns else 0
        render_type = chart_type

        if render_type == "Text":
            st.write(f"**Result:** {chart_data.iloc[0, 0]:.2f}")
            logger.info("Rendered chart %d as Text result", idx)
        
        elif render_type == "Scatter":
            if second_metric and dimension in chart_data.columns:
                fig = px.scatter(
                    chart_data,
                    x=metric,
                    y=second_metric,
                    color=dimension,
                    labels={metric: metric, second_metric: second_metric, dimension: dimension}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_scatter")
                logger.info("Rendered chart %d as Scatter chart", idx)
            else:
                st.warning("Scatter plot requires two metrics and a dimension. Falling back to Bar chart.")
                logger.warning("Second metric not found for scatter plot in chart %d, falling back to Bar", idx)
                if dimension in chart_data.columns:
                    unique_values = len(chart_data[dimension].unique())
                    if unique_values > 25 and selected_chart_type != "Table":
                        render_type = "Table"
                        st.warning(f"Switched to Table because the dimension '{dimension}' has {unique_values} unique values (> 25).")
                        logger.info("Switched chart %d to Table due to %d unique values in dimension %s (Scatter fallback)", idx, unique_values, dimension)
                    elif unique_values < 5 and selected_chart_type != "Table":
                        render_type = "Pie"
                        st.info(f"Switched to Pie chart because the dimension '{dimension}' has {unique_values} unique values (< 5).")
                        logger.info("Switched chart %d to Pie due to %d unique values in dimension %s (Scatter fallback)", idx, unique_values, dimension)
                    if render_type == "Bar":
                        fig = px.bar(
                            chart_data,
                            x=dimension,
                            y=metric,
                            labels={dimension: dimension, metric: metric}
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_bar_fallback")
                        logger.info("Rendered chart %d as Bar chart (fallback from Scatter)", idx)
                    elif render_type == "Pie":
                        fig = px.pie(
                            chart_data,
                            names=dimension,
                            values=metric,
                            labels={dimension: dimension, metric: metric}
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_pie_fallback")
                        logger.info("Rendered chart %d as Pie chart (fallback from Scatter)", idx)
                else:
                    st.error("Cannot render chart: Dimension not found in chart data.")
                    logger.error("Dimension not found for fallback chart in chart %d", idx)
                    return
        
        elif render_type == "Line":
            if secondary_dimension and secondary_dimension in chart_data.columns:
                fig = px.line(
                    chart_data,
                    x=dimension,
                    y=metric,
                    color=secondary_dimension,
                    labels={dimension: dimension, metric: metric, secondary_dimension: secondary_dimension}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_line")
                logger.info("Rendered chart %d as Line chart with secondary dimension", idx)
            elif dimension in chart_data.columns:
                fig = px.line(
                    chart_data,
                    x=dimension,
                    y=metric,
                    labels={dimension: dimension, metric: metric}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_line")
                logger.info("Rendered chart %d as Line chart", idx)
            else:
                st.error("Cannot render line chart: Dimension not found in chart data.")
                logger.error("Dimension not found for line chart in chart %d", idx)
                return
        
        elif render_type == "Bar":
            if secondary_dimension and secondary_dimension in chart_data.columns:
                fig = px.bar(
                    chart_data,
                    x=dimension,
                    y=metric,
                    color=secondary_dimension,
                    labels={dimension: dimension, metric: metric, secondary_dimension: secondary_dimension}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_bar")
                logger.info("Rendered chart %d as Bar chart with secondary dimension", idx)
            elif dimension in chart_data.columns:
                fig = px.bar(
                    chart_data,
                    x=dimension,
                    y=metric,
                    labels={dimension: dimension, metric: metric}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_bar")
                logger.info("Rendered chart %d as Bar chart", idx)
            else:
                st.error("Cannot render bar chart: Dimension not found in chart data.")
                logger.error("Dimension not found for bar chart in chart %d", idx)
                return
        
        elif render_type == "Pie":
            if dimension in chart_data.columns:
                fig = px.pie(
                    chart_data,
                    names=dimension,
                    values=metric,
                    labels={dimension: dimension, metric: metric}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_pie")
                logger.info("Rendered chart %d as Pie chart", idx)
            else:
                st.error("Cannot render pie chart: Dimension not found in chart data.")
                logger.error("Dimension not found for pie chart in chart %d", idx)
                return
        
        elif render_type == "Map" and dimension in chart_data.columns:
            if "Outlier_Label" in chart_data.columns:
                fig = px.choropleth(
                    chart_data,
                    locations=dimension,
                    locationmode="country names",
                    color="Outlier_Label",
                    hover_data=[metric],
                    color_discrete_map={"Outlier": "red", "Normal": "blue"},
                    labels={dimension: dimension, metric: metric}
                )
            else:
                fig = px.choropleth(
                    chart_data,
                    locations=dimension,
                    locationmode="country names",
                    color=metric,
                    hover_data=[metric],
                    labels={dimension: dimension, metric: metric}
                )
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_map")
            logger.info("Rendered chart %d as Map chart", idx)
        
        elif render_type == "Table":
            st.write("Displaying data as a table:")
            st.dataframe(chart_data)
            logger.info("Rendered chart %d as Table", idx)
        
        else:
            st.error("Cannot render chart: Invalid chart type or insufficient data.")
            logger.error("Invalid chart type %s or insufficient data for chart %d", chart_type, idx)
            return
        
        with st.expander("Show Data"):
            st.write("Chart Data:")
            st.dataframe(chart_data)
            logger.info("Displayed raw chart data for chart %d", idx)
        
        if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
            stats = calculate_statistics(working_df, metric)
            if stats:
                logger.info("Calculated statistics for %s: %s", metric, stats)
                insights = generate_gpt_insights(stats, metric, prompt, chart_data, dimension, second_metric)
                st.write("### Insights")
                for insight in insights:
                    st.write(f"- {insight}")
                logger.info("Generated insights for chart %d: %s", idx, insights)
    
    except Exception as e:
        st.error(f"Error rendering chart {idx}: {str(e)}")
        logger.error("Error rendering chart %d: %s", idx, str(e))

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

# Tabs
if st.session_state.current_project:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🛠️ Field Editor", "📈 Dashboard", "📜 Executive Summary"])
    
    with tab1:
        st.subheader("📊 Data Management")
        
        st.markdown("### 📤 Upload Dataset")
        with st.container():
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="upload_csv_unique")
            if uploaded_file:
                try:
                    df = load_data(uploaded_file)
                    df = preprocess_dates(df)
                    st.session_state.dataset = df
                    df.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
                    st.session_state.classified = False
                    st.session_state.sample_prompts = []
                    st.session_state.used_sample_prompts = []
                    st.session_state.sample_prompt_pool = []
                    st.session_state.last_used_pool_index = 0
                    st.session_state.field_types = {}
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
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab.")
        else:
            df = st.session_state.dataset.copy()
            df = preprocess_dates(df)
            
            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
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
            ids = st.session_state.field_types.get("id", [])
            
            st.subheader("📊 Recent Charts")
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
                st.markdown("### Raw Data Preview")
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
                                second_metric = None
                                for col in measures:
                                    if col in chart_data.columns and col != metric:
                                        second_metric = col
                                        break
                                if stats:
                                    insights = generate_gpt_insights(stats, metric, prompt_text, chart_data, dimension, second_metric)
                                    latex_content.append(r"\subsubsection{Insights}")
                                    latex_content.append(r"\begin{itemize}")
                                    for insight in insights:
                                        latex_content.append(r"\item " + insight.replace("_", r"\_"))
                                    latex_content.append(r"\end{itemize}")

                        latex_content.append(r"\end{document}")
                        latex_content = "\n".join(latex_content)

                        st.session_state[f"dashboard_pdf_content"] = latex_content
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
                        display_chart(idx, prompt_text, dimensions, measures, dates, df)
            
                with st.container():
                    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
                    user_prompt = st.text_input("💬 Ask about your data (e.g., 'Top 5 Cities by Sales' or 'Find outliers in Sales'):", key="manual_prompt")
                    st.markdown("#### 🧭 Available Fields")
                    st.markdown(f"**Dimensions:** {', '.join(dimensions)}")
                    st.markdown(f"**Measures:** {', '.join(measures)}")
                    st.markdown(f"**Dates:** {', '.join(dates) if dates else 'None'}")
                    
                    if st.button("🔎 Generate Chart", key="manual_prompt_button"):
                        if user_prompt:
                            st.session_state.last_manual_prompt = user_prompt
                            st.session_state.chart_history.append({"prompt": user_prompt})
                            logger.info("User generated chart with manual prompt: %s", user_prompt)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            
                if st.button("💾 Save Dashboard", key="save_dashboard"):
                    try:
                        save_dashboard(st.session_state.current_project, st.session_state.chart_history)
                        st.success("Dashboard saved successfully. View in the 'Executive Summary' tab.")
                        st.markdown('<div class="saved-dashboard">Dashboard Saved</div>', unsafe_allow_html=True)
                        logger.info("Saved dashboard for project: %s", st.session_state.current_project)
                    except Exception as e:
                        st.error(f"Failed to save dashboard: {str(e)}")
                        logger.error("Failed to save dashboard for project %s: %s", st.session_state.current_project, str(e))
    
    with tab4:
        st.subheader("📜 Executive Summary")
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab to view the executive summary.")
        else:
            df = st.session_state.dataset.copy()
            dimensions = st.session_state.field_types.get("dimension", [])
            measures = st.session_state.field_types.get("measure", [])
            dates = st.session_state.field_types.get("date", [])

            if st.button("📄 Export Executive Summary to PDF", key="export_summary_pdf"):
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

                st.session_state[f"summary_pdf_content"] = latex_content
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

            st.markdown("### 📝 Summary of Dashboard Analysis")
            if not st.session_state.chart_history:
                st.markdown("No analysis performed yet. Please generate charts in the 'Dashboard' tab to see a summary.")
            else:
                summary = generate_executive_summary(st.session_state.chart_history, df, dimensions, measures, dates)
                st.markdown("#### Key Findings and Recommendations")
                for point in summary:
                    st.markdown(f"- {point}")

            st.markdown("---")
            st.markdown("### 🔍 Overall Data Analysis and Findings")
            show_overall_analysis = st.toggle("Show Overall Data Analysis", value=False)
            if show_overall_analysis:
                overall_analysis = generate_overall_data_analysis(df, dimensions, measures, dates)
                st.markdown("#### Overall Insights")
                for point in overall_analysis:
                    st.markdown(f"- {point}")
else:
    st.info("Please select a project to continue.")