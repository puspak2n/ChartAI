import pandas as pd
import logging
import json
import os

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def classify_columns(df, existing_field_types=None):
    """
    Classify DataFrame columns into dimensions, measures, dates, and IDs.
    Returns a tuple of (dimensions, measures, dates, ids).
    """
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

        # Check for IDs (columns with "id" in name)
        if "id" in col.lower():
            ids.append(col)
            logger.info("Classified %s as ID (contains 'id' in name)", col)
            continue

        # Check for dates (datetime type or name contains "date")
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

        # Check for numeric types
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it could be a dimension (e.g., low unique value count)
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.05:  # Less than 5% unique values
                dimensions.append(col)
                logger.info("Classified %s as Dimension (numeric but low unique ratio=%.2f)", col, unique_ratio)
            else:
                measures.append(col)
                logger.info("Classified %s as Measure (numeric after conversion)", col)
            continue

        # Check for postal codes (numeric but should be dimension)
        if col.lower() in ["postal code", "zip code", "zip"]:
            dimensions.append(col)
            logger.info("Classified %s as Dimension (postal code)", col)
            continue

        # Check for categorical columns (object or string types)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.05:  # Less than 5% unique values
                dimensions.append(col)
                logger.info("Classified %s as Dimension (categorical, unique ratio=%.2f)", col, unique_ratio)
            else:
                dimensions.append(col)
                logger.info("Classified %s as Dimension (categorical, unique ratio=%.2f)", col, unique_ratio)

    logger.info("Final Classified columns - Dimensions: %s, Measures: %s, Dates: %s, IDs: %s", dimensions, measures, dates, ids)
    return dimensions, measures, dates, ids

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
    Load OpenAI API key from environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in environment.")
        return None
    logger.info("OpenAI API key loaded successfully.")
    return api_key

def generate_gpt_insight_with_fallback(chart_data, dimension, metric):
    """
    Generate insights without OpenAI (fallback since dependency is removed).
    """
    return []