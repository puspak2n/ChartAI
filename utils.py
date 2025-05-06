import pandas as pd
import logging
import json

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def classify_columns(df):
    """
    Classify DataFrame columns into dimensions, measures, dates, and IDs.
    Ensures mutual exclusivity between categories.
    """
    try:
        dimensions = []
        measures = []
        dates = []
        ids = []
        
        for col in df.columns:
            # Step 1: Check for IDs (columns containing "id" in the name, but exclude calculated fields)
            if "id" in col.lower() and not any(calc in col.lower() for calc in ["profit margin", "outliers", "indicator"]):
                ids.append(col)
                continue  # Skip to next column to ensure exclusivity
            
            # Step 2: Check for specific columns like Postal Code
            if col.lower() == "postal code":
                dimensions.append(col)
                continue  # Skip to next column to ensure exclusivity
            
            # Step 3: Check for dates
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dates.append(col)
                continue  # Skip to next column to ensure exclusivity
            
            # Step 4: Check for numeric columns (measures), but exclude IDs and Postal Code
            if pd.api.types.is_numeric_dtype(df[col]):
                measures.append(col)
                continue
            
            # Step 5: Default to dimensions for non-numeric, non-date, non-ID columns
            dimensions.append(col)
        
        # Remove any duplicates while preserving order
        dimensions = list(dict.fromkeys(dimensions))
        measures = list(dict.fromkeys(measures))
        dates = list(dict.fromkeys(dates))
        ids = list(dict.fromkeys(ids))
        
        # Ensure mutual exclusivity by removing IDs, dates, and specific columns from measures
        measures = [col for col in measures if col not in ids and col not in dates and col.lower() != "postal code"]
        dimensions = [col for col in dimensions if col not in ids and col not in dates and col not in measures]
        
        logger.info("Classified columns - Dimensions: %s, Measures: %s, Dates: %s, IDs: %s", dimensions, measures, dates, ids)
        return dimensions, measures, dates, ids
    except Exception as e:
        logger.error("Failed to classify columns: %s", str(e))
        return [], [], [], []

def load_data(uploaded_file):
    """
    Load data from an uploaded file.
    """
    try:
        df = pd.read_csv(uploaded_file)
        logger.info("Loaded dataset with %d rows and %d columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise

def save_dashboard(project_name, chart_history):
    """
    Save the dashboard configuration.
    """
    try:
        dashboard_data = {
            "charts": [{"prompt": chart["prompt"]} for chart in chart_history]
        }
        with open(f"projects/{project_name}/dashboard.json", "w") as f:
            json.dump(dashboard_data, f, indent=4)
        logger.info("Saved dashboard for project %s with %d charts", project_name, len(chart_history))
    except Exception as e:
        logger.error("Failed to save dashboard for project %s: %s", project_name, str(e))
        raise

def load_openai_key():
    """
    Load OpenAI API key from environment (placeholder since OpenAI dependency is removed).
    """
    logger.info("OpenAI key loading skipped as dependency is removed")
    return None

def generate_gpt_insight_with_fallback(chart_data, dimension, metric):
    """
    Generate insights without OpenAI (fallback since dependency is removed).
    """
    return []