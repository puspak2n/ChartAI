import re
import logging
import pandas as pd
import streamlit as st
import hashlib
import time
from utils import setup_logging
from calc_utils import detect_outliers
import plotly.express as px
from collections import OrderedDict
import openai
import json

logger = setup_logging()

# Define USE_OPENAI based on openai.api_key
USE_OPENAI = openai.api_key is not None

def find_column(name, candidates, df, exclude=[]):
    """Fuzzy match a column name from candidates."""
    name_lower = name.strip().lower()
    logger.info(f"Finding column for name: {name_lower} in candidates: {candidates}")
    
    # Handle common plural forms
    singular_forms = {
        'cities': 'city',
        'states': 'state',
        'countries': 'country',
        'regions': 'region',
        'categories': 'category',
        'sub-categories': 'sub-category',
        'products': 'product',
        'customers': 'customer',
        'segments': 'segment'
    }
    
    # Try singular form if the input is plural
    if name_lower in singular_forms:
        name_lower = singular_forms[name_lower]
    
    # First try exact match
    for col in candidates:
        if col.lower() == name_lower and col not in exclude:
            logger.info(f"Found exact match: {col}")
            return col
    
    # Then try contains match
    for col in candidates:
        if name_lower in col.lower() and col not in exclude:
            logger.info(f"Found contains match: {col}")
            return col
    
    # If no match found, log the failure and return None
    logger.warning(f"No match found for {name_lower} in candidates: {candidates}")
    return None

def find_date_column(name, date_candidates):
    for col in date_candidates:
        if name.lower() in col.lower():
            return col
    return date_candidates[0] if date_candidates else None

def detect_time_aggregation(name):
    if "year" in name.lower():
        return "year"
    elif "quarter" in name.lower():
        return "quarter"
    elif "month" in name.lower():
        return "month"
    else:
        return "month"

def parse_filter(filter_part, dimensions, measures, df):
    for col in dimensions + measures:
        if col.lower() in filter_part.lower():
            value = filter_part.lower().replace(col.lower(), "").replace("=", "").strip()
            unique_vals = df[col].dropna().astype(str).unique()
            for v in unique_vals:
                if value.lower() in str(v).lower():
                    return col, v
            return col, value
    return None, None

def rule_based_parse(prompt, df, dimensions, measures, dates):
    logger.info(f"Rule-based parsing prompt: {prompt}")
    logger.info(f"Available dimensions: {dimensions}")
    logger.info(f"Available measures: {measures}")
    prompt_lower = prompt.lower()
    chart_type = None
    metric = None
    dimension = None
    second_metric = None
    filter_col = None
    filter_val = None
    kwargs = {}
    is_two_metric = False
    exclude_list = []
    secondary_dimension = None

    patterns = OrderedDict([
        ("trend_over_time", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]*(?:date|month|year|quarter))\s*$"),
        ("trend_by_group", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]*(?:date|month|year|quarter))\s+and\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("compare_metrics", r"^\s*([a-zA-Z0-9_\s]+)\s+vs\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("top_n", r"^\s*top\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("bottom_n", r"^\s*bottom\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("top_n_filter", r"^\s*top\s+(\d+)?\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+(.+)$"),
        ("map_chart", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+country\s*$"),
        ("outliers", r"^\s*(?:show|find)\s+outliers\s+in\s+([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
        ("filter_category", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+([a-zA-Z0-9_\s]+)\s*=\s*(.+)$"),
        ("filter_value", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s+where\s+([a-zA-Z0-9_\s]+)\s*(>=|<=|>|<|=)\s*(\d+\.?\d*)\s*$"),
        ("basic_group", r"^\s*([a-zA-Z0-9_\s]+)\s+by\s+([a-zA-Z0-9_\s]+)\s*$"),
    ])

    for pattern_name, pattern in patterns.items():
        match = re.match(pattern, prompt_lower)
        if match:
            logger.info(f"Matched pattern: {pattern_name}")
            groups = match.groups()
            logger.info(f"Matched groups: {groups}")

            if pattern_name == "trend_over_time":
                metric_name, date_field = groups
                chart_type = "Line"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_date_column(date_field.strip(), dates)
                kwargs["time_aggregation"] = detect_time_aggregation(date_field)
                kwargs["sort_by_date"] = True

            elif pattern_name == "trend_by_group":
                metric_name, date_field, dim_name = groups
                chart_type = "Line"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_date_column(date_field.strip(), dates)
                secondary_dimension = find_column(dim_name.strip(), dimensions, df)
                kwargs["time_aggregation"] = detect_time_aggregation(date_field)
                kwargs["sort_by_date"] = True

            elif pattern_name == "compare_metrics":
                metric1_name, metric2_name, dim_name = groups
                chart_type = "Scatter"
                is_two_metric = True
                metric = find_column(metric1_name.strip(), measures, df)
                second_metric = find_column(metric2_name.strip(), measures, df, exclude=[metric])
                dimension = find_column(dim_name.strip(), dimensions, df)

            elif pattern_name in ["top_n", "bottom_n"]:
                n, dim_name, metric_name = groups
                logger.info(f"Processing top/bottom n: n={n}, dim_name={dim_name}, metric_name={metric_name}")
                chart_type = "Bar"
                n = int(n) if n else 5
                dimension = find_column(dim_name.strip(), dimensions, df)
                metric = find_column(metric_name.strip(), measures, df)

                if not dimension:
                    dimension = find_date_column(dim_name.strip(), dates)

                logger.info(f"Found dimension: {dimension}, metric: {metric}")
                if not dimension:
                    logger.error(f"Could not find dimension column for: {dim_name}")
                    return None
                if not metric:
                    logger.error(f"Could not find metric column for: {metric_name}")
                    return None

                if dim_name.strip().lower().endswith(("month", "year", "quarter")):
                    kwargs["time_aggregation"] = detect_time_aggregation(dim_name)
                    kwargs["sort_by_date"] = True

                kwargs["top_n"] = n
                kwargs["is_bottom"] = pattern_name == "bottom_n"

            elif pattern_name == "top_n_filter":
                n, dim_name, metric_name, filter_part = groups
                chart_type = "Bar"
                n = int(n) if n else 5
                dimension = find_column(dim_name.strip(), dimensions, df)
                metric = find_column(metric_name.strip(), measures, df)
                if not dimension or not metric:
                    return None
                kwargs["top_n"] = n
                filter_col, filter_val = parse_filter(filter_part, dimensions, measures, df)

            elif pattern_name == "map_chart":
                metric_name = groups[0]
                chart_type = "Map"
                dimension = "Country"
                metric = find_column(metric_name.strip(), measures, df)
                if "Country" not in dimensions:
                    chart_type = "Bar"
                    logger.warning("Country not in dimensions, fallback to Bar chart.")

            elif pattern_name == "outliers":
                metric_name, dim_name = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                if not dimension or not metric:
                    return None
                kwargs["show_outliers"] = True

            elif pattern_name == "filter_category":
                metric_name, dim_name, filter_col_name, filter_val_raw = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                filter_col = find_column(filter_col_name.strip(), dimensions, df)
                if not dimension or not metric or not filter_col:
                    return None
                filter_val = filter_val_raw.strip()

            elif pattern_name == "filter_value":
                metric_name, dim_name, filter_col_name, operator, value = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions, df)
                filter_col = find_column(filter_col_name.strip(), dimensions, df)
                if not dimension or not metric or not filter_col:
                    return None
                filter_val = f"{operator}{value}"

            elif pattern_name == "basic_group":
                metric_name, dim_name = groups
                chart_type = "Bar"
                metric = find_column(metric_name.strip(), measures, df)
                dimension = find_column(dim_name.strip(), dimensions + dates, df)
            
                if not metric or not dimension:
                    logger.warning(f"basic_group failed: metric={metric}, dimension={dimension}")
                    return None


            return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension

    if not chart_type:
        logger.warning("No pattern matched for prompt: %s", prompt)
        return None

    if not metric or not dimension:
        logger.error("Missing required components. Metric: %s, Dimension: %s", metric, dimension)
        return None

    logger.info("Parsed result: chart_type=%s, metric=%s, dimension=%s, second_metric=%s, filter_col=%s, filter_val=%s, kwargs=%s",
                chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs)
    return chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension

def render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """Render a chart based on the query and data."""
    try:
        logger.info(f"Starting chart rendering for prompt: {prompt}")
        
        # Parse the prompt to determine chart type and data requirements
        parsed_result = rule_based_parse(prompt, df, dimensions, measures, dates)
        if not parsed_result:
            raise ValueError("Failed to parse prompt")
            
        chart_type, metric, dimension, second_metric, filter_col, filter_val, kwargs, is_two_metric, exclude_list, secondary_dimension = parsed_result
        
        # Validate required components
        if not metric or not dimension:
            raise ValueError(f"Missing required components. Metric: {metric}, Dimension: {dimension}")
        
        # If no chart type was determined, use the provided one
        if not chart_type and chart_type is not None:
            chart_type = chart_type
        elif not chart_type:
            chart_type = "Bar"  # Default to Bar chart if no type specified
        
        # Prepare the data
        working_df = df.copy()

        if kwargs.get("show_outliers"):
            from calc_utils import detect_outliers
            try:
                # Add Outlier flags to original rows
                working_df = detect_outliers(working_df, metric, method="std")
                logger.info(f"Outliers detected: {working_df['Outlier'].value_counts().to_dict()}")
        
                # Save original for show_data
                original_df = working_df.copy()
        
                # Group data for visualization
                grouped = working_df.groupby([dimension_col, "Outlier"])[metric].sum().reset_index()
        
                # Ensure all dimension values are retained
                all_dims = working_df[[dimension_col]].drop_duplicates()
                all_dims["key"] = 1
                outlier_flags = pd.DataFrame({"Outlier": [True, False]})
                outlier_flags["key"] = 1
                dim_flag_combos = pd.merge(all_dims, outlier_flags, on="key").drop(columns="key")
        
                grouped = pd.merge(dim_flag_combos, grouped, on=[dimension_col, "Outlier"], how="left")
                grouped[metric].fillna(0, inplace=True)
                grouped["Color"] = grouped["Outlier"].map({True: "Outlier", False: "Normal"})
                kwargs["color_by"] = "Color"
                # Assign for chart
                working_df = grouped
        
                # Store both separately
                chart_df = grouped
                show_data_df = original_df[[dimension_col, metric, "Outlier"]]

        
                logger.info("Final grouped data prepared with outlier coloring and full category inclusion.")
            
            except Exception as e:
                logger.warning(f"Outlier detection failed: {e}")


        # Apply filter if specified

        if filter_col and filter_val is not None:
            try:
                working_df = working_df[
            working_df[filter_col].astype(str).str.lower().str.strip() == str(filter_val).lower().strip()
        ]
                logger.info(f"Applied filter: {filter_col} = {filter_val}, remaining rows: {len(working_df)}")
            except Exception as e:
                logger.warning(f"Failed to apply filter {filter_col} = {filter_val}: {e}")

        # Handle date dimensions
        time_agg = None
        if dimension in dates:
            working_df[dimension] = pd.to_datetime(working_df[dimension])
            # For time series, aggregate by date first
            if chart_type == "Line":
                # Determine time aggregation (month, quarter, year)
                time_agg = kwargs.get("time_aggregation", "month")
                if time_agg == "month":
                    working_df[dimension] = working_df[dimension].dt.to_period('M').dt.to_timestamp()
                elif time_agg == "quarter":
                    working_df[dimension] = working_df[dimension].dt.to_period('Q').dt.to_timestamp()
                elif time_agg == "year":
                    working_df[dimension] = working_df[dimension].dt.to_period('Y').dt.to_timestamp()
        
        # Group and aggregate the data
        if is_two_metric and second_metric:
            # For scatter plots comparing two metrics
            chart_data = working_df.groupby(dimension)[[metric, second_metric]].sum().reset_index()
            # Keep both metrics for scatter plot
            table_columns = [dimension, metric, second_metric]
        elif secondary_dimension:
            chart_data = working_df.groupby([dimension, secondary_dimension])[metric].sum().reset_index()
            table_columns = [dimension, secondary_dimension, metric]
        else:
            if "Outlier" in working_df.columns:
                chart_data = working_df.groupby([dimension, "Outlier"])[metric].sum().reset_index()
            else:
                chart_data = working_df.groupby(dimension)[metric].sum().reset_index()

            table_columns = [dimension, metric]
        
        # Handle top_n and bottom_n cases
        if "top_n" in kwargs:
            n = kwargs["top_n"]
            is_bottom = kwargs.get("is_bottom", False)
            # Sort based on whether it's top or bottom
            chart_data = chart_data.sort_values(by=metric, ascending=is_bottom)
            # Take top/bottom n
            chart_data = chart_data.head(n)
        
        # For time series, sort by date
        elif dimension in dates:
            chart_data = chart_data.sort_values(by=dimension)
        
        # For other cases, use the provided sort order
        elif sort_order == "Descending":
            chart_data = chart_data.sort_values(by=metric, ascending=False)
        else:
            chart_data = chart_data.sort_values(by=metric, ascending=True)
        
        # Add time aggregation to kwargs if it exists
        if time_agg:
            kwargs["time_aggregation"] = time_agg
        
        logger.info(f"Successfully rendered chart. Type: {chart_type}, Metric: {metric}, Dimension: {dimension}")
        return chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs
        
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}")
        raise ValueError(f"Failed to render chart: {str(e)}")

# In chart_utils.py
def generate_insights(chart_data, metric, dimension, secondary_dimension=None):
    """Generate insights from chart data using OpenAI API with a fallback to rule-based insights."""
    try:
        logger.info(f"Generating insights for chart: metric={metric}, dimension={dimension}, secondary_dimension={secondary_dimension}")

        # Check if OpenAI is available
        if USE_OPENAI and openai.api_key:
            try:
                # Calculate basic statistics for the prompt
                if metric in chart_data.columns and pd.api.types.is_numeric_dtype(chart_data[metric]):
                    stats = {
                        "mean": float(chart_data[metric].mean()),
                        "max": float(chart_data[metric].max()),
                        "min": float(chart_data[metric].min()),
                        "total": float(chart_data[metric].sum()),
                        "count": len(chart_data)
                    }
                else:
                    stats = {}

                # Prepare the prompt for OpenAI
                prompt = (
                    f"Generate 3 concise, actionable business insights for a data visualization:\n"
                    f"- Metric: {metric}\n"
                    f"- Dimension: {dimension}\n"
                    f"- Secondary Dimension: {secondary_dimension if secondary_dimension else 'None'}\n"
                    f"- Data Points: {len(chart_data)}\n"
                )
                if stats:
                    prompt += (
                        f"- Statistics: Mean={stats['mean']:.2f}, Max={stats['max']:.2f}, "
                        f"Min={stats['min']:.2f}, Total={stats['total']:.2f}\n"
                    )
                prompt += (
                    "Provide insights that highlight trends, key performers, or anomalies, "
                    "suitable for business decision-making. Format as a list of 3 bullet points."
                )

                # Call OpenAI API
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analyst providing concise, actionable insights from data visualizations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )

                # Extract and clean insights
                content = response.choices[0].message.content.strip()
                insights = [line.strip('- ').strip() for line in content.split('\n') if line.strip() and line.startswith('-')]
                insights = insights[:3]  # Ensure only 3 insights
                logger.info(f"Generated OpenAI insights: {insights}")
                return insights
            except Exception as e:
                logger.error(f"OpenAI API call failed for insights: {str(e)}")
                # Fall through to fallback mechanism

        # Fallback: Rule-based insights
        logger.info("Using fallback rule-based insights due to OpenAI unavailability or failure")
        insights = []

        if metric in chart_data.columns and pd.api.types.is_numeric_dtype(chart_data[metric]):
            mean_val = chart_data[metric].mean()
            max_val = chart_data[metric].max()
            min_val = chart_data[metric].min()
            total_val = chart_data[metric].sum()

            # Top performer insight
            top_performer = chart_data.loc[chart_data[metric].idxmax()]
            insights.append(f"Top performer: {top_performer[dimension]} with {metric} of {top_performer[metric]:.2f}")

            # Range insight
            insights.append(f"{metric} ranges from {min_val:.2f} to {max_val:.2f}, with an average of {mean_val:.2f}")

            # Total insight
            insights.append(f"Total {metric}: {total_val:.2f}")

            # Secondary dimension insights if available
            if secondary_dimension and secondary_dimension in chart_data.columns:
                top_by_secondary = chart_data.groupby(secondary_dimension)[metric].sum().sort_values(ascending=False)
                top_secondary = top_by_secondary.index[0]
                insights.append(f"Highest {metric} by {secondary_dimension}: {top_secondary} with {top_by_secondary.iloc[0]:.2f}")
                insights = insights[:3]  # Limit to 3 insights

            # Outlier detection
            q1 = chart_data[metric].quantile(0.25)
            q3 = chart_data[metric].quantile(0.75)
            iqr = q3 - q1
            outliers = chart_data[(chart_data[metric] < (q1 - 1.5 * iqr)) | (chart_data[metric] > (q3 + 1.5 * iqr))]
            if len(outliers) > 0:
                insights.append(f"Found {len(outliers)} potential outliers in the data")
                insights = insights[:3]  # Limit to 3 insights

        if not insights:
            insights.append("No significant insights could be generated from the data.")

        logger.info(f"Generated fallback insights: {insights}")
        return insights

    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return ["Unable to generate insights at this time."]