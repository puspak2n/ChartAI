import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
import openai
import logging
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

# Page Config
st.set_page_config(page_title="ChartGPT AI", page_icon="📊", layout="wide")

# Load Custom CSS and Override to Remove Excessive Top Spacing
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
</style>
""", unsafe_allow_html=True)

# Load OpenAI Key (no need to display status in production)
load_openai_key()

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
if "rerun_trigger" not in st.session_state:
    st.session_state.rerun_trigger = False
if "insights_cache" not in st.session_state:
    st.session_state.insights_cache = {}  # Cache for AI insights
if "sample_prompts" not in st.session_state:
    st.session_state.sample_prompts = []

# Sidebar
with st.sidebar:
    st.title("ChartGPT AI")
    
    # Projects Expander
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
        
        if selected_project != "(None)":
            try:
                if os.path.exists(f"projects/{selected_project}/dataset.csv"):
                    st.session_state.dataset = pd.read_csv(f"projects/{selected_project}/dataset.csv")
                    st.session_state.current_project = selected_project
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
                        st.success(f"Created: {new_project}")
                        logger.info("Created new project: %s", new_project)
                    except Exception as e:
                        st.error(f"Failed to create project: {e}")
                        logger.error("Failed to create project %s: %s", new_project, str(e))
                else:
                    st.error("Project already exists.")
                    logger.warning("Attempted to create project %s, but it already exists", new_project)
    
    # Sample Prompts Section (only if dataset is loaded)
    if st.session_state.dataset is not None:
        df = st.session_state.dataset.copy()
        for col in ['Order Date', 'Ship Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if "classified" not in st.session_state:
            try:
                dimensions, measures, dates, ids = classify_columns(df)
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
            if not st.session_state.sample_prompts:
                sample_prompts = generate_prompts_with_llm(dimensions, measures, dates, df)
                if not sample_prompts:
                    sample_prompts = generate_sample_prompts(dimensions, measures, dates, df)
                if measures:
                    sample_prompts.append(f"Find outliers in {measures[0]}")
                st.session_state.sample_prompts = sample_prompts[:5]  # Limit to 5 as per index.html
        
        st.markdown("### Sample Prompts")
        for idx, prompt in enumerate(st.session_state.sample_prompts):
            if st.button(prompt, key=f"sidebar_sample_{idx}"):
                st.session_state.chart_history.append({"prompt": prompt})
                st.rerun()
                logger.info("User selected sidebar sample prompt: %s", prompt)
    
    # About Expander
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **ChartGPT AI** is an AI-powered business intelligence platform that transforms data into actionable insights using natural language. Ask questions, visualize data, and uncover trends effortlessly.
        """)
    
    if not st.session_state.get("onboarding_seen", False):
        st.info("🚀 New to ChartGPT AI? Start with our guided onboarding to explore features!", icon="ℹ️")
        if st.button("Start Onboarding", key="onboarding_button"):
            st.session_state.onboarding_seen = True
            st.session_state.rerun_trigger = True
            logger.info("User started onboarding")

# Handle rerun trigger
if st.session_state.rerun_trigger:
    st.session_state.rerun_trigger = False
    st.rerun()

# Main Content
st.title("ChartGPT AI: Enterprise Insights")
if st.session_state.current_project:
    st.caption(f"Active Project: **{st.session_state.current_project}**")
else:
    st.warning("No project selected. Please create or open a project in the sidebar.")

# Onboarding Modal
if not st.session_state.get("onboarding_seen", False):
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
            st.session_state.rerun_trigger = True
            logger.info("User completed onboarding")

# Function to Save Dataset Changes
def save_dataset_changes():
    if st.session_state.current_project and st.session_state.dataset is not None:
        try:
            st.session_state.dataset.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
            st.session_state.pop("classified", None)
            logger.info("Saved dataset changes for project: %s", st.session_state.current_project)
        except Exception as e:
            st.error(f"Failed to save dataset: %s", str(e))
            logger.error("Failed to save dataset for project %s: %s", st.session_state.current_project, str(e))

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OpenAI API key not found in environment. Falling back to hard-coded insights.")
    USE_OPENAI = False
else:
    USE_OPENAI = True
    logger.info("OpenAI API key loaded successfully.")

def generate_gpt_insights(stats, metric, prompt, chart_data, dimension=None, second_metric=None):
    """
    Generate enhanced, diverse insights using OpenAI GPT, covering trends, comparisons, outliers, and balanced strategies.
    """
    if not USE_OPENAI:
        logger.info("Using hard-coded insights as OpenAI is not available.")
        return [
            f"The average {metric} is {stats['mean']:.2f}.",
            f"The standard deviation of {metric} is {stats['std_dev']:.2f}, indicating the spread of data.",
            f"The 25th percentile (Q1) of {metric} is {stats['q1']:.2f}, median is {stats['median']:.2f}, and 75th percentile (Q3) is {stats['q3']:.2f}.",
            f"The 90th percentile of {metric} is {stats['percentile_90']:.2f}, showing the top range."
        ]
    
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
        return [insight.strip('- ').strip() for insight in insights if insight.strip()]
    except Exception as e:
        logger.error("Failed to generate GPT insights: %s", str(e))
        return [
            f"The average {metric} is {stats['mean']:.2f}.",
            f"The standard deviation of {metric} is {stats['std_dev']:.2f}, indicating the spread of data.",
            f"The 25th percentile (Q1) of {metric} is {stats['q1']:.2f}, median is {stats['median']:.2f}, and 75th percentile (Q3) is {stats['q3']:.2f}.",
            f"The 90th percentile of {metric} is {stats['percentile_90']:.2f}, showing the top range."
        ]

def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    """
    Generate an executive summary based on the chart history prompts and their insights.
    """
    if not USE_OPENAI:
        logger.info("Using hard-coded executive summary as OpenAI is not available.")
        return [
            "Analysis Overview: Reviewed multiple aspects of the dataset.",
            "Key Finding: Sales performance varies significantly across categories.",
            "Recommendation: Focus on high-performing categories to maximize revenue."
        ]

    try:
        # Collect all insights from chart history
        all_insights = []
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result:
                chart_data, metric, dimension, working_df, table_columns, chart_type = chart_result
                stats = calculate_statistics(working_df, metric) if metric in working_df.columns else None
                second_metric = None
                for col in measures:
                    if col in chart_data.columns and col != metric:
                        second_metric = col
                        break
                if stats:
                    insights = generate_gpt_insights(stats, metric, prompt, chart_data, dimension, second_metric)
                    all_insights.append(f"**Prompt: {prompt}**\n" + "\n".join([f"- {insight}" for insight in insights]))

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

def generate_overall_data_analysis(df, dimensions, measures, dates):
    """
    Generate an overall data analysis and findings summary for the dataset.
    """
    if not USE_OPENAI:
        logger.info("Using hard-coded overall data analysis as OpenAI is not available.")
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

    try:
        # Gather key statistics for all measures
        stats_summary = []
        for metric in measures:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                stats = calculate_statistics(df, metric)
                stats_summary.append(
                    f"{metric}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}, "
                    f"Q1={stats['q1']:.2f}, median={stats['median']:.2f}, Q3={stats['q3']:.2f}, "
                    f"90th percentile={stats['percentile_90']:.2f}"
                )
        
        # Analyze top performers for key dimensions
        top_performers = []
        for dim in dimensions:
            for metric in measures:
                if dim in df.columns and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                    grouped = df.groupby(dim)[metric].mean().sort_values(ascending=False)
                    if not grouped.empty:
                        top = grouped.index[0]
                        top_value = grouped.iloc[0]
                        top_performers.append(f"Top {dim} by {metric}: {top} with average {top_value:.2f}")

        # Compute correlations between measures
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

def display_chart(idx, prompt, dimensions, measures, dates, df):
    """
    Display a chart with title on the left, chart type toggle and sort button on the right in the same row.
    """
    try:
        # Layout: Title on the left, controls on the right
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(prompt)

        with col2:
            chart_types = ["Bar", "Line", "Scatter", "Map", "Table"]
            default_index = 0
            if " vs " in prompt.lower():
                default_index = chart_types.index("Scatter")
            selected_chart_type = st.selectbox(
                "Chart Type",
                options=chart_types,
                index=default_index,
                key=f"chart_type_{idx}",
                label_visibility="collapsed"
            )

        with col3:
            sort_order = st.selectbox(
                "Sort Order",
                options=["Descending", "Ascending"],
                index=0,
                key=f"sort_order_{idx}",
                label_visibility="collapsed"
            )

        # Call render_chart with the selected chart type and sort order
        chart_result = render_chart(idx, prompt, dimensions, measures, dates, df, sort_order=sort_order, chart_type=selected_chart_type)
        if chart_result is None:
            st.error(f"Failed to render chart {idx} with prompt: {prompt}")
            logger.error("Chart result is None for prompt: %s", prompt)
            return
        
        chart_data, metric, dimension, working_df, table_columns, chart_type = chart_result
        
        # Define second_metric for insights generation
        second_metric = None
        for col in measures:
            if col in chart_data.columns and col != metric:
                second_metric = col
                break
        
        # Log chart data before processing
        logger.info("Chart data before processing: rows=%d, columns=%s", len(chart_data), chart_data.columns.tolist())
        
        # Render the chart based on chart_type
        if chart_type == "Scatter":
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
                    fig = px.bar(
                        chart_data,
                        x=dimension,
                        y=metric,
                        labels={dimension: dimension, metric: metric}
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_bar_fallback")
                    logger.info("Rendered chart %d as Bar chart (fallback from Scatter)", idx)
                else:
                    st.error("Cannot render chart: Dimension not found in chart data.")
                    logger.error("Dimension not found for fallback Bar chart in chart %d", idx)
                    return
        
        elif chart_type == "Line" and dimension in chart_data.columns:
            fig = px.line(
                chart_data,
                x=dimension,
                y=metric,
                labels={dimension: dimension, metric: metric}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_line")
            logger.info("Rendered chart %d as Line chart", idx)
        
        elif chart_type == "Bar" and dimension in chart_data.columns:
            fig = px.bar(
                chart_data,
                x=dimension,
                y=metric,
                labels={dimension: dimension, metric: metric}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}_bar")
            logger.info("Rendered chart %d as Bar chart", idx)
        
        elif chart_type == "Map" and dimension in chart_data.columns:
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
        
        elif chart_type == "Table":
            st.write("Displaying data as a table:")
            st.dataframe(chart_data)
            logger.info("Rendered chart %d as Table", idx)
        
        else:
            st.error("Cannot render chart: Invalid chart type or insufficient data.")
            logger.error("Invalid chart type %s or insufficient data for chart %d", chart_type, idx)
            return
        
        # Show data toggle using an expander
        with st.expander("Show Data"):
            st.write("Chart Data:")
            st.dataframe(chart_data)
            logger.info("Displayed raw chart data for chart %d", idx)
        
        # Generate enhanced insights
        if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
            stats = calculate_statistics(working_df, metric)
            logger.info("Calculated statistics for %s: %s", metric, stats)
            insights = generate_gpt_insights(stats, metric, prompt, chart_data, dimension, second_metric)
            st.write("### Insights")
            for insight in insights:
                st.write(f"- {insight}")
            logger.info("Generated insights for chart %d: %s", idx, insights)
    
    except Exception as e:
        st.error(f"Error rendering chart {idx}: {str(e)}")
        logger.error("Error rendering chart %d: %s", idx, str(e))

if st.session_state.current_project:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📤 Data Upload", "🛠️ Field Editor", "🔍 Data Explorer", "📜 Executive Summary"])
    
    with tab1:
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data Upload' tab.")
        else:
            df = st.session_state.dataset.copy()
            for col in ['Order Date', 'Ship Date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            if "classified" not in st.session_state:
                try:
                    dimensions, measures, dates, ids = classify_columns(df)
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
            
            if not measures or not dimensions:
                st.error("Dataset must have at least one numeric (measure) and one categorical (dimension) column.")
                logger.error("Dataset lacks required columns: measures=%s, dimensions=%s", measures, dimensions)
                st.stop()
            
            st.markdown("#### 🧭 Available Fields")
            st.markdown(f"**Dimensions:** {', '.join(dimensions)}")
            st.markdown(f"**Measures:** {', '.join(measures)}")
            st.markdown(f"**Dates:** {', '.join(dates) if dates else 'None'}")
            st.markdown(f"**All Columns:** {', '.join(df.columns)}")
            
            st.subheader("📊 Recent Charts")
            if len(st.session_state.chart_history) > 10:
                st.session_state.chart_history = st.session_state.chart_history[-5:]
            
            if not st.session_state.chart_history:
                st.markdown("No recent charts to display.")
            else:
                for idx, chart_obj in enumerate(st.session_state.chart_history):
                    prompt_text = chart_obj["prompt"]
                    display_chart(idx, prompt_text, dimensions, measures, dates, df)
            
            with st.container():
                st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
                user_prompt = st.text_input("💬 Ask about your data (e.g., 'Top 5 Cities by Sales' or 'Find outliers in Sales'):", key="manual_prompt")
                if st.button("🔎 Generate Chart", key="manual_prompt_button"):
                    if user_prompt:
                        st.session_state.chart_history.append({"prompt": user_prompt})
                        st.rerun()
                        logger.info("User generated chart with manual prompt: %s", user_prompt)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("💾 Save Dashboard", key="save_dashboard"):
                try:
                    save_dashboard(st.session_state.current_project, st.session_state.chart_history)
                    st.success("Dashboard saved successfully. View in the 'Executive Summary' tab.")
                    st.markdown('<div class="saved-dashboard">Dashboard Saved</div>', unsafe_allow_html=True)
                    logger.info("Saved dashboard for project: %s", st.session_state.current_project)
                except Exception as e:
                    st.error(f"Failed to save dashboard: %s", str(e))
                    logger.error("Failed to save dashboard for project %s: %s", st.session_state.current_project, str(e))
    
    with tab2:
        if st.session_state.current_project:
            st.subheader("📊 Add Data to Project")
            uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], key="upload_csv_unique")
            if uploaded_file:
                try:
                    df = load_data(uploaded_file)
                    for col in ['Order Date', 'Ship Date']:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    st.session_state.dataset = df
                    df.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
                    st.success("✅ Dataset uploaded!")
                    st.session_state.rerun_trigger = True
                    logger.info("Uploaded dataset for project: %s", st.session_state.current_project)
                except Exception as e:
                    st.error(f"Failed to upload dataset: %s", str(e))
                    logger.error("Failed to upload dataset for project %s: %s", st.session_state.current_project, str(e))
    
    with tab3:
        if st.session_state.dataset is not None:
            st.subheader("🛠️ Field Editor")
            df = st.session_state.dataset

            st.markdown("### 🔧 Manage Fields and Types")
            with st.expander("Manage Fields and Types"):
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
                                st.success(f"Field renamed to {new_name}!")
                                st.session_state.rerun_trigger = True
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
                            st.success(f"Field {col} deleted!")
                            st.session_state.rerun_trigger = True
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
                st.markdown("""
                Examples:
                - "Mark Sales as High if greater than 1000, otherwise Low"
                - "Calculate the profit margin as Profit divided by Sales"
                - "Flag outliers in Sales where Sales is more than 2 standard deviations above the average"
                - "Calculate average Profit per Ship Mode and flag if above overall average"
                - "If Sales is greater than 500 and Profit is positive, then High Performer, else if Sales is less than 200, then Low Performer, else Medium"
                """)
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
                        if input_mode == "Prompt-based (Plain English)":
                            formula = generate_formula_from_prompt(
                                calc_prompt,
                                st.session_state.field_types.get("dimension", []),
                                st.session_state.field_types.get("measure", []),
                                df
                            )
                        else:
                            formula = formula_input
                        
                        if formula:
                            for col in df.columns:
                                formula = formula.replace(f"[{col}]", col)
                            
                            st.markdown(f"**Formula Used:** `{formula}`")
                            result = evaluate_calculation(formula, df.copy(), group_by=group_by)
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
                                st.success(f"New field {new_field_name} created!")
                                st.session_state.rerun_trigger = True
                                logger.info("Created new calculated field %s with formula: %s", new_field_name, formula)
    
    with tab4:
        if st.session_state.dataset is not None:
            st.subheader("🔍 Data Explorer")
            df = st.session_state.dataset
            st.markdown("### 📄 Dataset Preview (Top 100 Rows)")
            st.dataframe(df.head(100))
            
            st.markdown("### 🧭 Unique Values by Dimension")
            dimensions = st.session_state.field_types.get("dimension", [])
            for dim in dimensions:
                unique_vals = df[dim].dropna().unique()[:10]
                st.markdown(f"**{dim}**: {', '.join(map(str, unique_vals))}")
    
    with tab5:
        st.subheader("📜 Executive Summary")
        if st.session_state.dataset is None:
            st.info("No dataset loaded. Please upload a dataset in the 'Data Upload' tab to view the executive summary.")
        else:
            df = st.session_state.dataset.copy()
            dimensions = st.session_state.field_types.get("dimension", [])
            measures = st.session_state.field_types.get("measure", [])
            dates = st.session_state.field_types.get("date", [])

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