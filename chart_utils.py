import pandas as pd
import plotly.express as px
import numpy as np
import logging
import re
from calc_utils import detect_outliers

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def rule_based_parse(prompt, dimensions, measures, dates, df):
    """
    Parse a natural language prompt to extract metrics, dimensions, and other chart parameters.
    Returns a tuple: (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension)
    """
    prompt_lower = prompt.lower().strip()
    
    # Initialize default values
    metric = None
    dimension = None
    second_metric = None
    trend = None
    top_n = None
    filters = {}
    map_request = False
    table_columns = []
    secondary_dimension = None
    
    # Helper to check if a column exists in measures or dimensions
    def find_column(token, columns):
        for col in columns:
            if col.lower() == token.lower():
                return col
        return None
    
    # Helper to check for calculated measures
    def find_calculated_measure(token):
        if "profit margin" in token.lower():
            return "Calculate the profit margin as Profit divided by Sales"
        return token
    
    # Step 1: Check for scatter plot requests (e.g., "scatter plot with sales vs profit with each country")
    if "scatter plot with" in prompt_lower and "with each" in prompt_lower:
        scatter_match = re.search(r'scatter plot with (.+?) vs (.+?) with each (.+)', prompt_lower)
        if scatter_match:
            metric_part = scatter_match.group(1).strip()
            second_metric_part = scatter_match.group(2).strip()
            dimension_part = scatter_match.group(3).strip()
            metric = find_column(metric_part, measures)
            second_metric = find_column(second_metric_part, measures)
            dimension = find_column(dimension_part, dimensions)
            logger.info("Parsed scatter plot prompt: metric=%s, second_metric=%s, dimension=%s", metric, second_metric, dimension)
            return (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension)
    
    # Step 2: Check for scatter plot requests in the form "[metric] vs [second_metric] by [dimension]"
    if "vs" in prompt_lower and "by" in prompt_lower:
        scatter_match = re.search(r'(.+?) vs (.+?) by (.+)', prompt_lower)
        if scatter_match:
            metric_part = scatter_match.group(1).strip()
            second_metric_part = scatter_match.group(2).strip()
            dimension_part = scatter_match.group(3).strip()
            metric = find_column(metric_part, measures)
            second_metric = find_column(second_metric_part, measures)
            dimension = find_column(dimension_part, dimensions)
            if metric and second_metric and dimension:
                logger.info("Parsed scatter plot prompt (vs/by format): metric=%s, second_metric=%s, dimension=%s", metric, second_metric, dimension)
                return (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension)
    
    # Step 3: Check for map request (e.g., "by country")
    if "by country" in prompt_lower:
        map_request = True
        dimension = "Country"
        logger.info("Parsed prompt (by country): metric=%s, dimension=Country, map_request=True", metric)
    
    # Step 4: Check for trend requests (e.g., "trend over")
    if "trend over" in prompt_lower:
        tokens = prompt_lower.split("trend over")
        if len(tokens) > 1:
            trend_part = tokens[1].strip()
            for date_col in dates:
                if date_col.lower() in trend_part:
                    trend = date_col
                    dimension = date_col  # Set dimension to the date column for trend analysis
                    break
            # Extract metric before "trend over"
            metric_part = tokens[0].strip()
            for measure in measures:
                if measure.lower() in metric_part:
                    metric = measure
                    break
            # Extract secondary dimension after "by"
            if "by" in trend_part:
                dim_part = trend_part.split("by")[1].strip()
                for dim in dimensions:
                    if dim.lower() in dim_part and dim != dimension:
                        secondary_dimension = dim
                        break
        logger.info("Parsed prompt: metric=%s, dimension=%s, trend=%s, secondary_dimension=%s", metric, dimension, trend, secondary_dimension)
    
    # Step 5: Check for comparison (e.g., "compare X and Y by Z")
    if "compare" in prompt_lower and "and" in prompt_lower:
        tokens = prompt_lower.split("compare")[1].split("and")
        if len(tokens) >= 2:
            metric1_part = tokens[0].strip()
            metric2_part = tokens[1].split("by")[0].strip()
            for measure in measures:
                if measure.lower() in metric1_part:
                    metric = measure
                    break
            for measure in measures:
                if measure.lower() in metric2_part:
                    second_metric = measure
                    break
            # Extract dimension after "by"
            if "by" in prompt_lower:
                dim_part = prompt_lower.split("by")[1].strip()
                for dim in dimensions:
                    if dim.lower() in dim_part:
                        dimension = dim
                        break
    
    # Step 6: Check for top N requests (e.g., "top 5 X by Y")
    if "top" in prompt_lower:
        tokens = prompt_lower.split()
        for i, token in enumerate(tokens):
            if token == "top" and i + 1 < len(tokens):
                try:
                    top_n = int(tokens[i + 1])
                    # Extract dimension after "top N"
                    dim_part = " ".join(tokens[i + 2:]).split("by")[0].strip()
                    for dim in dimensions:
                        if dim.lower() in dim_part.lower():
                            dimension = dim
                            break
                    # Extract metric after "by"
                    if "by" in prompt_lower:
                        metric_part = prompt_lower.split("by")[1].strip()
                        for measure in measures:
                            if measure.lower() in metric_part:
                                metric = measure
                                break
                except ValueError:
                    pass
        # Check for implicit filters (e.g., "for Customer Name")
        for dim in dimensions:
            if f"for {dim.lower()}" in prompt_lower:
                value = prompt_lower.split(f"for {dim.lower()}")[1].strip().split()[0]
                filters[dim] = ("=", value)
                logger.info("Extracted implicit filter: %s = %s", dim, value)
    
    # Step 7: Check for outlier requests (e.g., "find outliers in Sales by Sub-Category")
    if "find outliers in" in prompt_lower:
        filters["outliers"] = True
        tokens = prompt_lower.split("find outliers in")[1].strip()
        # Extract metric
        for measure in measures:
            if measure.lower() in tokens.lower():
                metric = measure
                break
        # Extract dimension after "by"
        if "by" in tokens:
            dim_part = tokens.split("by")[1].strip()
            for dim in dimensions:
                if dim.lower() in dim_part:
                    dimension = dim
                    break
        else:
            # If no dimension is specified, default to the first dimension
            dimension = dimensions[0] if dimensions else None
            logger.info("No dimension specified for outlier request, defaulting to %s", dimension)
        logger.info("Parsed prompt: metric=%s, dimension=%s (outlier request)", metric, dimension)
        return (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension)
    
    # Step 8: Default parsing (e.g., "Sales by Category")
    if not metric and not dimension:
        tokens = prompt_lower.split("by")
        if len(tokens) > 1:
            metric_part = tokens[0].strip()
            dim_part = tokens[1].strip()
            metric = find_calculated_measure(metric_part)
            for measure in measures:
                if measure.lower() in metric.lower():
                    metric = measure
                    break
            for dim in dimensions:
                if dim.lower() in dim_part:
                    dimension = dim
                    break
    
    logger.info("Parsed prompt: metric=%s, dimension=%s", metric, dimension)
    logger.info("Parsed prompt: %s", (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension))
    
    # Validate that metric and dimension are identified
    if not metric or (not dimension and not trend and not filters.get("outliers")):
        logger.error("Metric or dimension not identified for prompt: %s", prompt)
        return None
    
    return (metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension)

def render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """
    Render a chart based on the parsed prompt.
    Returns a tuple: (chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension)
    """
    # Parse the prompt
    parse_result = rule_based_parse(prompt, dimensions, measures, dates, df)
    if parse_result is None:
        return None
    
    metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns, secondary_dimension = parse_result
    
    # Prepare working DataFrame
    working_df = df.copy()
    logger.info("Before filtering: rows=%d", len(working_df))
    
    # Apply filters
    for col, (op, value) in filters.items():
        if col == "outliers":
            continue
        if op == "=":
            working_df = working_df[working_df[col] == value]
            logger.info("Applied filter: %s = %s, rows=%d", col, value, len(working_df))
    
    logger.info("After filtering: rows=%d", len(working_df))
    
    # Handle calculated metrics
    if "profit margin" in metric.lower():
        working_df["Profit Margin"] = working_df["Profit"] / working_df["Sales"]
        metric = "Profit Margin"
        logger.info("Computed Profit Margin as Profit / Sales")
    
    # Aggregate data
    chart_data = working_df.copy()
    
    # Handle outlier detection
    if filters.get("outliers"):
        if metric and dimension:
            try:
                # Ensure detect_outliers returns a DataFrame with the expected columns
                outlier_df = detect_outliers(chart_data, metric, dimension)
                if isinstance(outlier_df, pd.DataFrame) and 'Outlier' in outlier_df.columns:
                    chart_data = outlier_df
                    chart_data["Outlier_Label"] = chart_data["Outlier"].apply(lambda x: "Outlier" if x else "Normal")
                else:
                    logger.error("detect_outliers returned invalid data for metric=%s, dimension=%s", metric, dimension)
                    return None
            except Exception as e:
                logger.error("Error in detect_outliers for metric=%s, dimension=%s: %s", metric, dimension, str(e))
                return None
        else:
            try:
                outlier_df = detect_outliers(chart_data, metric)
                if isinstance(outlier_df, pd.DataFrame) and 'Outlier' in outlier_df.columns:
                    chart_data = outlier_df
                    chart_data["Outlier_Label"] = chart_data["Outlier"].apply(lambda x: "Outlier" if x else "Normal")
                else:
                    logger.error("detect_outliers returned invalid data for metric=%s", metric)
                    return None
            except Exception as e:
                logger.error("Error in detect_outliers for metric=%s: %s", metric, str(e))
                return None
    
    # Aggregate by dimension if specified
    if dimension and not filters.get("outliers"):
        agg_columns = [dimension]
        if secondary_dimension:
            agg_columns.append(secondary_dimension)
        if trend:
            agg_columns = [trend] + ([secondary_dimension] if secondary_dimension else [])
        # Aggregate both metric and second_metric if present
        agg_metrics = [metric]
        if second_metric:
            agg_metrics.append(second_metric)
        chart_data = chart_data.groupby(agg_columns)[agg_metrics].mean().reset_index()
        logger.info("Aggregating metric(s) %s by dimension(s) %s", agg_metrics, agg_columns)
    
    # Apply top N if specified
    if top_n:
        chart_data = chart_data.nlargest(top_n, metric)
        logger.info("Applied top_n=%d: rows=%d", top_n, len(chart_data))
    
    # Sort the data
    if sort_order == "Ascending":
        chart_data = chart_data.sort_values(by=metric, ascending=True)
    else:
        chart_data = chart_data.sort_values(by=metric, ascending=False)
    logger.info("Applied %s sorting on chart_data by %s: rows=%d", sort_order, metric, len(chart_data))
    
    # Determine chart type if not specified
    if not chart_type:
        if trend:
            chart_type = "Line"
        elif second_metric:
            chart_type = "Scatter"
        elif map_request:
            chart_type = "Map"
        elif filters.get("outliers"):
            chart_type = "Scatter"  # Use scatter plot for outliers
        else:
            chart_type = "Bar"
    logger.info("Selected chart type for chart %d: %s (default was %s)", idx, chart_type, chart_type)
    
    logger.info("Returning chart_data: rows=%d, columns=%s", len(chart_data), chart_data.columns.tolist())
    return (chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension)
