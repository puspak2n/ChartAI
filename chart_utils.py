import pandas as pd
import plotly.express as px
import logging
import re

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from calc_utils import detect_outliers

def rule_based_parse(prompt, dimensions, measures, dates):
    """
    Parse a natural language prompt into chart parameters using rule-based logic.
    Returns a tuple of (metric, dimension, second_metric, trend, top_n, filters, outliers, exclude, secondary_dimension).
    """
    try:
        prompt_lower = prompt.lower().strip()
        logger.info("Parsed prompt: %s", prompt_lower)
        
        # Initialize variables
        metric = None
        dimension = None
        second_metric = None
        trend = None
        top_n = None
        filters = {}
        outliers = False
        exclude = []
        secondary_dimension = None
        map_request = False
        
        # Check for map requests (e.g., "by country")
        if "by country" in prompt_lower:
            map_request = True
            dimension = next((dim for dim in dimensions if dim.lower() == "country"), None)
            logger.info("Parsed prompt (by country): metric=%s, dimension=%s, map_request=%s", metric, dimension, map_request)
        
        # Extract top N (e.g., "top 5 cities")
        top_n_match = re.search(r'top (\d+)', prompt_lower)
        if top_n_match:
            top_n = int(top_n_match.group(1))
            # Look for the dimension after "top N"
            top_n_dim_match = re.search(r'top \d+ (.+?)(?:\s+by|\s+for|\s*$)', prompt_lower)
            if top_n_dim_match:
                potential_dim = top_n_dim_match.group(1).strip()
                for dim in dimensions:
                    if dim.lower() == potential_dim:
                        dimension = dim
                        break
        
        # Check for trend keywords
        if "trend" in prompt_lower or "over" in prompt_lower:
            # Look for date columns in the prompt
            for date_col in dates:
                if date_col.lower() in prompt_lower:
                    trend = date_col
                    dimension = date_col
                    break
        
        # Check for "by order month" or "by order year"
        if "by order month" in prompt_lower and "Order Date" in dates:
            dimension = "Order Month"
            logger.info("Detected 'by order month', setting dimension to 'Order Month'")
        if "by order year" in prompt_lower and "Order Date" in dates:
            dimension = "Order Year"
            logger.info("Detected 'by order year', setting dimension to 'Order Year'")
        
        # Identify metrics (measures)
        for measure in measures:
            measure_lower = measure.lower()
            pattern = rf'^{measure_lower}\b|\b{measure_lower}\b(?=.*\bby\b)'
            if re.search(pattern, prompt_lower):
                if not metric:
                    metric = measure
                elif not second_metric:
                    second_metric = measure
                else:
                    break
        
        # Relaxed search for metrics
        if not metric:
            for measure in measures:
                measure_lower = measure.lower()
                if measure_lower in prompt_lower:
                    if not metric:
                        metric = measure
                    elif not second_metric:
                        second_metric = measure
                    else:
                        break
        
        # Identify dimensions
        if not dimension:
            for date_col in dates:
                date_col_lower = date_col.lower()
                if f"by {date_col_lower}" in prompt_lower:
                    dimension = date_col
                    break
        
        if not dimension:
            for dim in dimensions:
                dim_lower = dim.lower()
                if f"by {dim_lower}" in prompt_lower:
                    if not dimension:
                        dimension = dim
                    elif not secondary_dimension:
                        secondary_dimension = dim
                    else:
                        break
        
        # Check for outliers
        if "outliers" in prompt_lower:
            outliers = True
        
        # Extract filters (e.g., "for Aaron Bergman")
        for dim in dimensions:
            dim_lower = dim.lower()
            if "for " in prompt_lower:
                filter_match = re.search(rf'for (.+?)(?:\s+by|\s*$)', prompt_lower)
                if filter_match:
                    filter_value = filter_match.group(1).strip()
                    filters[dim] = filter_value
                    break
        
        # Extract exclusions
        if "excluding" in prompt_lower:
            exclude_match = re.search(r'excluding (.+?)(?:\s+by|\s*$)', prompt_lower)
            if exclude_match:
                exclude_values = exclude_match.group(1).strip().split(',')
                exclude = [val.strip() for val in exclude_values]
        
        # Fallback dimension parsing
        if not dimension and not trend and not map_request:
            by_match = re.search(r'by (.+?)(?:\s+for|\s*$)', prompt_lower)
            if by_match:
                potential_dim = by_match.group(1).strip()
                for dim in dimensions + dates:
                    if dim.lower() == potential_dim:
                        dimension = dim
                        break
        
        # Fallback metric
        if not metric and measures:
            metric = measures[0]
            logger.info("No metric identified in prompt '%s', defaulting to first measure: %s", prompt_lower, metric)
        
        # Validation
        if not metric:
            logger.error("No metric identified and no measures available for prompt: %s", prompt_lower)
            return None, None, None, None, None, {}, False, [], None
        
        logger.info("Parsed prompt: metric=%s, dimension=%s", metric, dimension)
        logger.info("Parsed prompt: %s", (metric, dimension, second_metric, trend, top_n, filters, outliers, exclude, secondary_dimension))
        return metric, dimension, second_metric, trend, top_n, filters, outliers, exclude, secondary_dimension
    except Exception as e:
        logger.error("Failed to parse prompt %s: %s", prompt, str(e))
        return None, None, None, None, None, {}, False, [], None

def render_chart(chart_idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """
    Render a chart based on the parsed prompt.
    Returns a tuple of (chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension).
    """
    try:
        # Parse the prompt
        metric, dimension, second_metric, trend, top_n, filters, outliers, exclude, secondary_dimension = rule_based_parse(prompt, dimensions, measures, dates)
        
        if not metric or (not dimension and not trend and not second_metric):
            logger.error("Metric or dimension not identified for prompt: %s", prompt)
            return None
        
        chart_data = df.copy()
        working_df = df.copy()
        
        logger.info("Before filtering: rows=%d", len(chart_data))
        
        # Apply filters
        for dim, val in filters.items():
            if dim in chart_data.columns:
                chart_data = chart_data[chart_data[dim].str.lower() == val.lower()]
        
        # Apply exclusions
        for val in exclude:
            for col in chart_data.columns:
                chart_data = chart_data[chart_data[col].str.lower() != val.lower()]
        
        logger.info("After filtering: rows=%d", len(chart_data))
        
        # Handle "Order Month" and "Order Year"
        if dimension == "Order Month" and "Order Month" not in chart_data.columns:
            if "Order Date" in chart_data.columns:
                chart_data["Order Month"] = chart_data["Order Date"].dt.to_period("M").astype(str)
                working_df["Order Month"] = working_df["Order Date"].dt.to_period("M").astype(str)
                dimension = "Order Month"
                if "Order Month" not in dimensions:
                    dimensions.append("Order Month")
            else:
                logger.error("Order Date not found for Order Month calculation")
                return None
        
        if dimension == "Order Year" and "Order Year" not in chart_data.columns:
            if "Order Date" in chart_data.columns:
                chart_data["Order Year"] = chart_data["Order Date"].dt.year.astype(str)
                working_df["Order Year"] = working_df["Order Date"].dt.year.astype(str)
                dimension = "Order Year"
                if "Order Year" not in dimensions:
                    dimensions.append("Order Year")
            else:
                logger.error("Order Date not found for Order Year calculation")
                return None
        
        # Aggregate data
        agg_columns = []
        if metric:
            agg_columns.append(metric)
        if second_metric:
            agg_columns.append(second_metric)
        
        group_by = []
        if dimension:
            group_by.append(dimension)
        if secondary_dimension:
            group_by.append(secondary_dimension)
        
        if agg_columns and group_by:
            logger.info("Aggregating metric(s) %s by dimension(s) %s", agg_columns, group_by)
            chart_data = chart_data.groupby(group_by)[agg_columns].sum().reset_index()
        
        # Handle top N
        if top_n and metric:
            logger.info("Applied top_n=%d", top_n)
            chart_data = chart_data.nlargest(top_n, metric)
        
        # Apply sorting
        if metric:
            if sort_order == "Descending":
                chart_data = chart_data.sort_values(by=metric, ascending=False)
                logger.info("Applied Descending sorting on chart_data by %s: rows=%d", metric, len(chart_data))
            else:
                chart_data = chart_data.sort_values(by=metric, ascending=True)
                logger.info("Applied Ascending sorting on chart_data by %s: rows=%d", metric, len(chart_data))
        
        # Handle outlier detection
        if outliers:
            if metric and dimension:
                try:
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
        
        # Use user-selected chart type if provided, otherwise determine default
        if chart_type:
            render_type = chart_type
        else:
            if trend:
                render_type = "Line"
            elif second_metric:
                render_type = "Scatter"
            elif "by country" in prompt.lower():
                render_type = "Map"
            elif " vs " in prompt.lower():
                render_type = "Scatter"
            else:
                render_type = "Bar"
        
        logger.info("Selected chart type for chart %d: %s (user-selected: %s)", chart_idx, render_type, chart_type)
        
        # Prepare table columns for display
        table_columns = []
        if dimension:
            table_columns.append(dimension)
        if secondary_dimension:
            table_columns.append(secondary_dimension)
        if metric:
            table_columns.append(metric)
        if second_metric:
            table_columns.append(second_metric)
        if "Outlier_Label" in chart_data.columns:
            table_columns.append("Outlier_Label")
        
        logger.info("Returning chart_data: rows=%d, columns=%s", len(chart_data), chart_data.columns.tolist())
        return chart_data, metric, dimension, working_df, table_columns, render_type, secondary_dimension
    except Exception as e:
        logger.error("Error rendering chart %d: %s", chart_idx, str(e))
        return None