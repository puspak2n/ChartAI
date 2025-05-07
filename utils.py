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

def classify_columns(df, field_types=None):
    """
    Classify DataFrame columns into dimensions, measures, dates, and IDs.
    Respects existing classifications in field_types if provided.
    Ensures at least one measure and one dimension are identified for flexibility.
    """
    try:
        # Initialize categories
        dimensions = []
        measures = []
        dates = []
        ids = []
        
        # Use existing field_types if provided, otherwise initialize empty lists
        if field_types:
            dimensions = field_types.get("dimension", []).copy()
            measures = field_types.get("measure", []).copy()
            dates = field_types.get("date", []).copy()
            ids = field_types.get("id", []).copy()
        
        # Log the initial DataFrame structure for debugging
        logger.info("Dataset columns: %s", df.columns.tolist())
        logger.info("Dataset dtypes: %s", df.dtypes.to_dict())
        logger.info("Existing field_types: %s", field_types)
        
        # Columns that have already been classified
        classified_columns = dimensions + measures + dates + ids
        unclassified_columns = [col for col in df.columns if col not in classified_columns]
        
        for col in unclassified_columns:
            # Step 1: Check for dates
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dates.append(col)
                logger.info("Classified %s as Date (datetime type)", col)
                continue
            
            # Step 2: Check for IDs (columns containing "id" in the name)
            if "id" in col.lower():
                ids.append(col)
                logger.info("Classified %s as ID (contains 'id' in name)", col)
                continue
            
            # Step 3: Check for specific columns like Postal Code (treat as dimension)
            if col.lower() == "postal code":
                dimensions.append(col)
                logger.info("Classified %s as Dimension (postal code)", col)
                continue
            
            # Step 4: Check for numeric columns (measures)
            # First, attempt to convert the column to numeric to catch string-based numbers
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().mean() > 0.5:  # At least 50% of values are numeric
                    measures.append(col)
                    logger.info("Classified %s as Measure (numeric after conversion)", col)
                    continue
            except Exception as e:
                logger.debug("Could not convert %s to numeric: %s", col, str(e))
            
            # If pandas already identifies it as numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                measures.append(col)
                logger.info("Classified %s as Measure (numeric dtype)", col)
                continue
            
            # Step 5: Treat as dimension if non-numeric, non-date, non-ID
            # Check if the column has a reasonable number of unique values to be a categorical dimension
            unique_count = df[col].nunique()
            total_count = len(df[col])
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            # Relaxed heuristic: consider columns with unique ratio < 0.8 or unique count < 100 as dimensions
            if unique_ratio < 0.8 or unique_count < 100:
                dimensions.append(col)
                logger.info("Classified %s as Dimension (categorical, unique ratio=%.2f)", col, unique_ratio)
            else:
                logger.info("Column %s has too many unique values (ratio=%.2f), not classified as dimension", col, unique_ratio)
        
        # Remove any duplicates while preserving order
        dimensions = list(dict.fromkeys(dimensions))
        measures = list(dict.fromkeys(measures))
        dates = list(dict.fromkeys(dates))
        ids = list(dict.fromkeys(ids))
        
        # Ensure mutual exclusivity: remove IDs and dates from measures and dimensions
        measures = [col for col in measures if col not in ids and col not in dates]
        dimensions = [col for col in dimensions if col not in ids and col not in dates and col not in measures]
        
        # Fallback: If no measures or dimensions are found, force classify to ensure usability
        unclassified = [col for col in df.columns if col not in measures and col not in dimensions and col not in dates and col not in ids]
        
        if not measures and unclassified:
            # Prefer numeric columns for measures
            for col in unclassified:
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().any():  # At least some values are numeric
                        measures.append(col)
                        logger.info("Fallback: Classified %s as Measure (forced numeric)", col)
                        if col in dimensions:
                            dimensions.remove(col)
                        break
                except:
                    continue
            
            # If still no measures, take the first unclassified column
            if not measures and unclassified:
                col = unclassified[0]
                measures.append(col)
                logger.info("Fallback: Classified %s as Measure (forced, no numeric found)", col)
                if col in dimensions:
                    dimensions.remove(col)
        
        if not dimensions and unclassified:
            # Prefer non-numeric columns for dimensions, or columns with low unique value counts
            for col in unclassified:
                if col not in measures:
                    unique_count = df[col].nunique()
                    total_count = len(df[col])
                    unique_ratio = unique_count / total_count if total_count > 0 else 0
                    dimensions.append(col)
                    logger.info("Fallback: Classified %s as Dimension (forced, unique ratio=%.2f)", col, unique_ratio)
                    break
            
            # If still no dimensions, take the first unclassified column not in measures
            if not dimensions:
                for col in unclassified:
                    if col not in measures:
                        dimensions.append(col)
                        logger.info("Fallback: Classified %s as Dimension (forced, no categorical found)", col)
                        break
        
        logger.info("Final Classified columns - Dimensions: %s, Measures: %s, Dates: %s, IDs: %s", dimensions, measures, dates, ids)
        return dimensions, measures, dates, ids
    except Exception as e:
        logger.error("Failed to classify columns: %s", str(e))
        # Fallback to a minimal classification
        unclassified = df.columns.tolist()
        if unclassified:
            return [unclassified[0]], [unclassified[0]], [], []
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
