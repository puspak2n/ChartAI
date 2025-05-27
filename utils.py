import pandas as pd
import logging
import json
import os
import uuid
from supabase import create_client
import re

def setup_logging():
    """Configure logging for the entire application."""
    if not logging.getLogger().handlers:  # Only configure if no handlers exist
        logging.basicConfig(
            filename="chartgpt.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize Supabase client
try:
    supabase = create_client("https://fyyvfaqiohdxhnbdqoxu.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5eXZmYXFpb2hkeGhuYmRxb3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NTA2MTYsImV4cCI6MjA2MzEyNjYxNn0.-h6sm3bgPzxDjxlmPhi5LNzsbhMJiz8-0HX80U7FiZc")
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    raise


def classify_columns(df, existing_field_types=None):
    if existing_field_types is None:
        existing_field_types = {}
    dimensions = existing_field_types.get("dimension", [])
    measures = existing_field_types.get("measure", [])
    dates = existing_field_types.get("date", [])
    ids = existing_field_types.get("id", [])
    logger.info("Dataset columns: %s", list(df.columns))
    logger.info("Dataset dtypes: %s", {col: df[col].dtype for col in df.columns})
    logger.info("Existing field_types: %s", existing_field_types)
    for col in df.columns:
        if col in dimensions or col in measures or col in dates or col in ids:
            continue
        if "id" in col.lower():
            ids.append(col)
            logger.info("Classified %s as ID (contains 'id' in name)", col)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dates.append(col)
            logger.info("Classified %s as Date (datetime type)", col)
            continue
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                dates.append(col)
                logger.info("Explicitly converted %s to datetime due to 'date' in name", col)
                continue
            except Exception as e:
                logger.debug("Could not convert %s to datetime: %s", col, str(e))
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.05:
                dimensions.append(col)
                logger.info("Classified %s as Dimension (numeric but low unique ratio=%.2f)", col, unique_ratio)
            else:
                measures.append(col)
                logger.info("Classified %s as Measure (numeric after conversion)", col)
            continue
        if col.lower() in ["postal code", "zip code", "zip"]:
            dimensions.append(col)
            logger.info("Classified %s as Dimension (postal code)", col)
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col])
            dimensions.append(col)
            logger.info("Classified %s as Dimension (categorical, unique ratio=%.2f)", col, unique_ratio)
    logger.info("Final Classified columns - Dimensions: %s, Measures: %s, Dates: %s, IDs: %s", dimensions, measures, dates, ids)
    return dimensions, measures, dates, ids

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise

def generate_unique_id():
    return str(uuid.uuid4())

def log_session_state(session_state, context=""):
    """Log session state keys and token excerpts."""
    logger.info(f"Session state {context} keys: {list(session_state.keys())}")
    logger.info(f"Stored access_token {context}: {session_state.get('access_token', 'None')[:10]}...")
    logger.info(f"Stored refresh_token {context}: {session_state.get('refresh_token', 'None')[:10]}...")

def save_dashboard(supabase, project_id, dashboard_name, charts, user_id, session_state):
    logger.info(f"Attempting to save dashboard: name={dashboard_name}, project_id={project_id}, user_id={user_id}, charts={charts}")
    dashboard_id = generate_unique_id()
    try:
        log_session_state(session_state, "before saving dashboard")
        logger.info(f"Session state keys: {list(session_state.keys())}")
        logger.info(f"Stored access_token: {session_state.get('access_token', 'None')[:10]}...")
        logger.info(f"Stored refresh_token: {session_state.get('refresh_token', 'None')[:10]}...")

        # Verify and reapply session
        session = supabase.auth.get_session()
        if not session:
            if "access_token" in session_state and "refresh_token" in session_state:
                logger.info("Attempting to refresh session with stored refresh token")
                supabase.auth.set_session(session_state["access_token"], session_state["refresh_token"])
                session = supabase.auth.get_session()
                if not session:
                    logger.error("Failed to reapply Supabase session")
                else:
                    logger.info(f"Session reapplied: access_token={session.access_token[:10] + '...'}")
            else:
                logger.warning("No active Supabase session or stored tokens, proceeding with current client state")
        else:
            logger.info(f"Supabase session active: access_token={session.access_token[:10] + '...'}")

        # Deduplicate charts
        unique_charts = []
        seen = set()
        for chart in charts:
            chart_tuple = (chart.get("prompt", ""), chart.get("chart_type", ""))
            if chart_tuple not in seen:
                seen.add(chart_tuple)
                unique_charts.append(chart)
            else:
                logger.info(f"Removed duplicate chart: prompt={chart.get('prompt')}, chart_type={chart.get('chart_type')}")
        logger.info(f"Deduplicated charts: original={len(charts)}, unique={len(unique_charts)}")

        # Insert dashboard
        response = supabase.table("dashboards").insert({
            "id": dashboard_id,
            "project_id": project_id,
            "name": dashboard_name,
            "charts": unique_charts,
            "owner_id": user_id
        }).execute()
        logger.info(f"Inserted dashboard: response={response.data}")

        # Insert permission (handle failure gracefully)
        try:
            response = supabase.table("permissions").insert({
                "dashboard_id": dashboard_id,
                "user_id": user_id,
                "role": "Admin"
            }).execute()
            logger.info(f"Inserted permission: response={response.data}")
        except Exception as e:
            logger.error(f"Failed to insert permission: {str(e)}")

        logger.info(f"Saved dashboard '{dashboard_name}' (ID: {dashboard_id}) for project: {project_id}, user: {user_id}")
        return dashboard_id
    except Exception as e:
        logger.error(f"Error saving dashboard: {str(e)}")
        return None

def load_dashboards(supabase, user_id, session_state):
    logger.info(f"Loading dashboards for user_id: {user_id}")
    try:
        log_session_state(session_state, "before loading dashboards")
        logger.info(f"Session state keys: {list(session_state.keys())}")
        logger.info(f"Stored access_token: {session_state.get('access_token', 'None')[:10]}...")
        logger.info(f"Stored refresh_token: {session_state.get('refresh_token', 'None')[:10]}...")

        # Verify and reapply session
        session = supabase.auth.get_session()
        if not session:
            if "access_token" in session_state and "refresh_token" in session_state:
                logger.info("Attempting to refresh session with stored refresh token")
                supabase.auth.set_session(session_state["access_token"], session_state["refresh_token"])
                session = supabase.auth.get_session()
                if not session:
                    logger.error("Failed to reapply Supabase session")
                else:
                    logger.info(f"Session reapplied: access_token={session.access_token[:10] + '...'}")
            else:
                logger.warning("No active Supabase session or stored tokens, proceeding with current client state")
        else:
            logger.info(f"Supabase session active: access_token={session.access_token[:10] + '...'}")

        response = supabase.table("dashboards").select("*").or_(f"owner_id.eq.{user_id},id.in.(select dashboard_id from permissions where user_id = {user_id})").execute()
        dashboards = pd.DataFrame(response.data)
        logger.info(f"Loaded dashboards for user {user_id}: {len(dashboards)} rows, data={dashboards.to_dict()}")
        return dashboards
    except Exception as e:
        logger.error(f"Error loading dashboards: {e}")
        return pd.DataFrame()

def save_annotation(project_name, dashboard_id, chart_prompt, annotation):
    try:
        os.makedirs(f"projects/{project_name}", exist_ok=True)
        annotation_file = f"projects/{project_name}/annotations.json"
        annotation_data = {
            "dashboard_id": str(dashboard_id),
            "chart_prompt": chart_prompt if chart_prompt else "",
            "annotation": annotation,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        existing_data = []
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        existing_data.append(annotation_data)
        with open(annotation_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        logger.info("Saved annotation for dashboard %s, chart '%s' in project: %s", dashboard_id, chart_prompt, project_name)
    except Exception as e:
        logger.error("Failed to save annotation for project %s: %s", project_name, str(e))
        raise

def load_annotations(project_name):
    try:
        annotation_file = f"projects/{project_name}/annotations.json"
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            df = pd.DataFrame(data)
            logger.info("Loaded annotations for project: %s, rows=%d", project_name, len(df))
            return df
        return pd.DataFrame(columns=["dashboard_id", "chart_prompt", "annotation", "created_at"])
    except Exception as e:
        logger.error("Failed to load annotations for project %s: %s", project_name, str(e))
        return pd.DataFrame(columns=["dashboard_id", "chart_prompt", "annotation", "created_at"])

def delete_dashboard(project_name, dashboard_id):
    try:
        dashboard_file = f"projects/{project_name}/dashboard.json"
        if os.path.exists(dashboard_file):
            with open(dashboard_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            data = [d for d in data if isinstance(d, dict) and str(d["dashboard_id"]) != str(dashboard_id)]
            with open(dashboard_file, "w") as f:
                json.dump(data, f, indent=4)
            logger.info("Deleted dashboard %s from project: %s", dashboard_id, project_name)
        annotation_file = f"projects/{project_name}/annotations.json"
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            if not isinstance(annotations, list):
                annotations = [annotations]
            annotations = [a for a in annotations if str(a["dashboard_id"]) != str(dashboard_id)]
            with open(annotation_file, "w") as f:
                json.dump(annotations, f, indent=4)
            logger.info("Deleted annotations for dashboard %s in project: %s", dashboard_id, project_name)
    except Exception as e:
        logger.error("Failed to delete dashboard %s for project %s: %s", dashboard_id, project_name, str(e))
        raise

def parse_prompt(prompt, dimensions, measures, dates):
    """
    Parse a prompt into components for chart rendering.
    Returns: (metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension)
    """
    # This function is deprecated. Use rule_based_parse from chart_utils.py instead.
    logger.warning("Using deprecated parse_prompt function. Please use rule_based_parse from chart_utils.py instead.")
    from chart_utils import rule_based_parse
    return rule_based_parse(prompt, None, dimensions, measures, dates)

def update_dashboard(project_name, dashboard_id, new_name=None, new_prompts=None):
    try:
        dashboard_file = f"projects/{project_name}/dashboard.json"
        if not os.path.exists(dashboard_file):
            logger.warning("No dashboard file found for project: %s", project_name)
            return
        with open(dashboard_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for d in data:
            if isinstance(d, dict) and str(d["dashboard_id"]) == str(dashboard_id):
                if new_name:
                    d["dashboard_name"] = new_name
                if new_prompts:
                    d["charts"] = [{"prompt": p} for p in new_prompts]
                d["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        with open(dashboard_file, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Updated dashboard %s in project: %s", dashboard_id, project_name)
    except Exception as e:
        logger.error("Failed to update dashboard %s for project %s: %s", dashboard_id, project_name, str(e))
        raise

def load_openai_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in environment.")
        return None
    logger.info("OpenAI API key loaded successfully.")
    return api_key

def generate_gpt_insight_with_fallback(chart_data, dimension, metric):
    return []