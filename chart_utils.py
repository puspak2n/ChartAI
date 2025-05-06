import pandas as pd
import plotly.express as px
import re
import logging
import uuid
import random
import openai
import os
from calc_utils import detect_outliers, evaluate_calculation

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load OpenAI API key for prompt generation
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OpenAI API key not found for prompt generation. Falling back to rule-based generation.")
    USE_OPENAI = False
else:
    USE_OPENAI = True
    logger.info("OpenAI API key loaded for prompt generation.")

def rule_based_parse(prompt, dimensions, measures, df):
    """
    Parse a natural language prompt to extract visualization components.
    Returns: (metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns)
    """
    prompt_lower = prompt.lower().strip()
    metric = None
    second_metric = None
    dimension = None
    trend = None
    top_n = None
    filters = {}
    map_request = False
    table_columns = []
    
    # Handle "as a table" in prompt
    if "as a table" in prompt_lower:
        table_columns = list(df.columns)
    elif "a table with" in prompt_lower:
        table_match = re.search(r'a table with (.+?)(?=\s*(?:with filter|$))', prompt_lower)
        if table_match:
            columns_part = table_match.group(1).strip()
            requested_columns = [col.strip() for col in columns_part.split(',')]
            table_columns = []
            for col in requested_columns:
                for df_col in df.columns:
                    if col.lower() == df_col.lower():
                        table_columns.append(df_col)
                        break
            logger.info("Parsed table columns: %s", table_columns)
    
    # Check for trend over time (support "trend of ... over time", "trend over", "trend by")
    if "trend" in prompt_lower:
        # Look for "trend of <metric> over time" pattern
        trend_match = re.search(r'trend of (\w+\s*\w*) over time', prompt_lower)
        if trend_match:
            metric_part = trend_match.group(1)
            for col in measures:
                if col.lower() == metric_part.lower():
                    metric = col
                    break
            # Look for "by <dimension>"
            by_match = re.search(r'by (\w+\s*\w*)', prompt_lower)
            if by_match:
                dimension_part = by_match.group(1)
                for col in dimensions:
                    if col.lower() == dimension_part.lower():
                        dimension = col
                        break
            # Default trend to a date column
            trend = dates[0] if dates else None
    
    # Check for top N
    top_n_match = re.search(r'top (\d+)', prompt_lower)
    if top_n_match:
        top_n = int(top_n_match.group(1))
    
    # Check for map request
    if "country" in prompt_lower:
        map_request = True
    
    # Extract filters (e.g., "with filter Segment = Consumer")
    if "with filter" in prompt_lower:
        filter_match = re.search(r'with filter (.+?)(?=\s*(?:with filter|$))', prompt_lower)
        if filter_match:
            where_clause = filter_match.group(1).strip()
            logger.info("Extracted where clause: %s", where_clause)
            field_value = re.match(r'(\w+\s*\w*)\s*=\s*([\w\s]+)', where_clause)
            if field_value:
                field, value = field_value.groups()
                for col in df.columns:
                    if col.lower() == field.lower():
                        field = col
                        break
                filters[field] = ("=", value.strip())
                logger.info("Parsed filter: field=%s, operator==, value=%s", field, value)
    
    # Handle "compare ... across ..." or "vs" prompts
    if "compare" in prompt_lower and "across" in prompt_lower:
        compare_match = re.search(r'compare (.+?) across (?:different )?(\w+\s*\w*)', prompt_lower)
        if compare_match:
            metric_part = compare_match.group(1).strip()
            dimension_part = compare_match.group(2).strip()
            # Check for calculated metric (e.g., "profit margin (Profit divided by Sales)")
            calc_metric_match = re.search(r'(\w+\s*\w*)\s*\((.+?)\)', metric_part)
            if calc_metric_match:
                metric_name = calc_metric_match.group(1).strip()
                expression = calc_metric_match.group(2).strip()
                # Replace column names in the expression
                for col in df.columns:
                    if col.lower() in expression.lower():
                        expression = expression.replace(col.lower(), col)
                metric = expression  # e.g., "Profit / Sales"
            else:
                for col in measures:
                    if col.lower() in metric_part.lower():
                        metric = col
                        break
            for col in dimensions:
                if col.lower() == dimension_part.lower():
                    dimension = col
                    break
    
    # Handle "vs" prompts (e.g., "Profit vs Sales by Country")
    if " vs " in prompt_lower:
        parts = prompt_lower.split(" vs ")
        first_part = parts[0].strip()
        second_part = parts[1].strip()
        second_parts = second_part.split(" by ")
        dimension_part = None
        if len(second_parts) >= 2:
            metric_part = first_part
            second_metric_part = second_parts[0].strip()
            dimension_part = second_parts[1].strip()
        elif "with each" in second_part:
            metric_part = first_part
            second_metric_part = second_parts[0].replace("with each", "").strip()
            dimension_part = second_part.split("with each")[-1].strip()
        else:
            metric_part = first_part
            second_metric_part = second_part
        for col in measures:
            if col.lower() in metric_part:
                metric = col
                break
        for col in measures:
            if col.lower() == second_metric_part:
                second_metric = col
                break
        if dimension_part:
            for col in dimensions:
                if col.lower() == dimension_part:
                    dimension = col
                    break
        logger.info("Parsed prompt: metric=%s, dimension=%s, second_metric=%s, trend=%s, top_n=%s, filters=%s, map_request=%s, table_columns=%s",
                    metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns)
        return metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns
    
    # Handle "and" prompts (e.g., "Sales and Profit by Segment")
    if " and " in prompt_lower and " by " in prompt_lower:
        parts = prompt_lower.split(" by ")
        first_part = parts[0].strip()
        dimension_part = parts[1].strip()
        metric_parts = first_part.split(" and ")
        if len(metric_parts) == 2:
            metric_part = metric_parts[0].strip()
            second_metric_part = metric_parts[1].strip()
            for col in measures:
                if col.lower() in metric_part:
                    metric = col
                    break
            for col in measures:
                if col.lower() in second_metric_part:
                    second_metric = col
                    break
            for col in dimensions:
                if col.lower() == dimension_part:
                    dimension = col
                    break
            logger.info("Parsed prompt: metric=%s, dimension=%s, second_metric=%s, trend=%s, top_n=%s, filters=%s, map_request=%s, table_columns=%s",
                        metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns)
            return metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns
    
    # Handle "scatter plot with ... vs ... with each ..." prompts
    if "scatter plot with" in prompt_lower and "with each" in prompt_lower:
        scatter_match = re.search(r'scatter plot with (.+?) vs (.+?) with each (.+)', prompt_lower)
        if scatter_match:
            metric_part = scatter_match.group(1).strip()
            second_metric_part = scatter_match.group(2).strip()
            dimension_part = scatter_match.group(3).strip()
            for col in measures:
                if col.lower() == metric_part:
                    metric = col
                    break
            for col in measures:
                if col.lower() == second_metric_part:
                    second_metric = col
                    break
            for col in dimensions:
                if col.lower() == dimension_part:
                    dimension = col
                    break
            logger.info("Parsed prompt: metric=%s, dimension=%s, second_metric=%s, trend=%s, top_n=%s, filters=%s, map_request=%s, table_columns=%s",
                        metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns)
            return metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns
    
    # Handle "by" prompts (e.g., "Discount by Country", "Top 5 Country by Discount")
    if " by " in prompt_lower:
        parts = prompt_lower.split(" by ")
        first_part = parts[0].strip()
        second_part = parts[1].strip()
        second_part = re.sub(r'with filter (.+)', '', second_part).strip()
        if top_n:
            dimension_part = first_part.replace(f"top {top_n}", "").strip()
            metric_part = second_part
            for col in dimensions:
                if col.lower() == dimension_part:
                    dimension = col
                    break
            for col in measures:
                if col.lower() == metric_part:
                    metric = col
                    break
        else:
            metric_part = first_part
            dimension_part = second_part
            for col in measures:
                if col.lower() in metric_part:
                    metric = col
                    logger.info("Matched multi-word metric: %s", metric)
                    break
            if not metric:
                for col in measures:
                    if col.lower() in metric_part.split():
                        metric = col
                        break
            for col in dimensions:
                if col.lower() == dimension_part.lower():
                    dimension = col
                    break
    
    # Handle outlier prompts (e.g., "Find outliers in Sales")
    if "outlier" in prompt_lower:
        outlier_match = re.search(r'find outliers in (\w+)', prompt_lower)
        if outlier_match:
            metric = outlier_match.group(1)
            for col in measures:
                if col.lower() == metric.lower():
                    metric = col
                    break
            # Look for dimension in "by <dimension>"
            by_match = re.search(r'by (\w+\s*\w*)', prompt_lower)
            if by_match:
                dimension_part = by_match.group(1)
                for col in dimensions:
                    if col.lower() == dimension_part.lower():
                        dimension = col
                        break
    
    # Fallback: if no metric found, try to find a measure in the prompt
    if not metric:
        for col in measures:
            if col.lower() in prompt_lower.split():
                metric = col
                break
    
    # Fallback: if no dimension found, try to find a dimension in the prompt
    if not dimension and not trend:
        for col in dimensions:
            if col.lower() in prompt_lower.split():
                dimension = col
                break
    
    logger.info("Parsed prompt: metric=%s, dimension=%s, second_metric=%s, trend=%s, top_n=%s, filters=%s, map_request=%s, table_columns=%s",
                metric, dimension, second_metric, trend, top_n, filters, map_request, table_columns)
    return metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns

def generate_sample_prompts(dimensions, measures, dates, df):
    """
    Generate a list of sample prompts using GPT for smarter, varied prompts, with a fallback to rule-based.
    """
    if USE_OPENAI:
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            dimensions_str = ", ".join(dimensions)
            measures_str = ", ".join(measures)
            dates_str = ", ".join(dates)
            
            unique_values = {}
            for dim in dimensions:
                if dim in df.columns:
                    unique_vals = df[dim].dropna().unique()
                    if len(unique_vals) > 0:
                        unique_values[dim] = unique_vals[0]
            unique_values_str = ", ".join([f"{k}={v}" for k, v in unique_values.items()])
            
            prompt = (
                f"Generate 5 concise, insightful, and varied natural language prompts for data visualization. "
                f"Available columns - Dimensions: {dimensions_str}. Measures: {measures_str}. Dates: {dates_str}. "
                f"Unique values for filters: {unique_values_str}. "
                f"Include a mix of: "
                f"- Trend analysis (e.g., sales trends over time by a dimension), "
                f"- Comparisons (e.g., comparing two metrics across a dimension), "
                f"- Top N rankings (e.g., top 5 categories by profit), "
                f"- Filtered views (e.g., sales by region for a specific customer), "
                f"- Outlier detection or correlations (e.g., find outliers in profit by category). "
                f"Ensure prompts are actionable and relevant for business insights."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst creating insightful visualization prompts for business users."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            prompts = response.choices[0].message.content.strip().split('\n')
            prompts = [p.strip('- ').strip() for p in prompts if p.strip()]
            logger.info("Generated GPT-based sample prompts: %s", prompts)
            return prompts[:5]
        except Exception as e:
            logger.error("Failed to generate GPT-based prompts: %s", str(e))
            logger.warning("Falling back to rule-based prompt generation.")

    prompts = []
    for measure in measures:
        for dimension in dimensions:
            prompts.append(f"{measure} by {dimension}")
            break
    for measure in measures:
        for dimension in dimensions:
            prompts.append(f"Top 5 {dimension} by {measure}")
            break
    for measure in measures:
        for date in dates:
            prompts.append(f"{measure} trend over {date}")
            prompts.append(f"{measure} trend by {date}")
            break
    unique_values = {}
    for dimension in dimensions:
        if dimension in df.columns:
            unique_vals = df[dimension].dropna().unique()
            if len(unique_vals) > 0:
                unique_values[dimension] = unique_vals[0]
    if "Customer Name" in unique_values:
        for measure in measures:
            for dimension in dimensions:
                prompts.append(f"{measure} by {dimension} with filter Customer Name = {unique_values['Customer Name']}")
                break
    if len(measures) >= 2:
        for dimension in dimensions:
            prompts.append(f"{measures[0]} vs {measures[1]} by {dimension}")
            break
    if len(dimensions) >= 2:
        for measure in measures:
            prompts.append(f"{measure} by {dimensions[0]} and {dimensions[1]}")
            break
    if len(measures) >= 2:
        prompts.append(f"Correlation between {measures[0]} and {measures[1]}")
    for measure in measures:
        for dimension in dimensions:
            prompts.append(f"Find outliers in {measure} by {dimension}")
            break
    if len(measures) >= 2 and len(dimensions) >= 1:
        prompts.append(f"a table with {measures[0]}, {dimensions[0]} and {measures[1]}")
    random.shuffle(prompts)
    logger.info("Generated rule-based sample prompts: %s", prompts)
    return prompts[:5]

def render_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """
    Render a chart based on the parsed prompt.
    """
    try:
        logger.info("Rendering chart %d with prompt: %s", idx, prompt)
        logger.info("Received sort_order: %s, chart_type: %s", sort_order, chart_type)
        
        # Parse the prompt
        metric, second_metric, dimension, trend, top_n, filters, map_request, table_columns = rule_based_parse(prompt, dimensions, measures, df)
        
        if not metric:
            logger.error("Could not identify a metric in the prompt: %s", prompt)
            return None
        
        # Prepare working DataFrame
        working_df = df.copy()
        
        # Convert date columns to datetime if trend is specified
        if trend:
            if trend not in working_df.columns:
                logger.error("Trend column %s not found in DataFrame", trend)
                return None
            
            logger.info("Raw %s values before any conversion: %s", trend, working_df[trend].head().tolist())
            logger.info("Sample %s values before conversion: %s", trend, working_df[trend].head().tolist())
            
            date_formats = [
                "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y",
                "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y", "%d.%m.%Y"
            ]
            parsed_dates = None
            for date_format in date_formats:
                try:
                    parsed_dates = pd.to_datetime(working_df[trend], format=date_format, errors='coerce')
                    invalid_dates = parsed_dates.isna().sum()
                    if invalid_dates < len(working_df):
                        logger.info("Successfully parsed %s with format %s; %d invalid dates", trend, date_format, invalid_dates)
                        break
                except Exception as e:
                    logger.debug("Failed to parse %s with format %s: %s", trend, date_format, str(e))
            
            if parsed_dates is not None and not parsed_dates.isna().all():
                working_df[trend] = parsed_dates
            else:
                try:
                    parsed_dates = pd.to_datetime(working_df[trend], errors='coerce')
                    logger.info("Fallback to pandas default parsing for %s", trend)
                    working_df[trend] = parsed_dates
                except Exception as e:
                    logger.error("Failed to parse %s even with default parsing: %s", trend, str(e))
                    return None
            
            invalid_dates = working_df[trend].isna().sum()
            if invalid_dates > 0:
                logger.warning("Dropped %d rows with invalid dates in %s", invalid_dates, trend)
                working_df = working_df.dropna(subset=[trend])
            
            if working_df.empty:
                logger.error("DataFrame is empty after dropping invalid dates in %s", trend)
                other_date_cols = [col for col in dates if col != trend]
                if other_date_cols:
                    logger.info("Suggestion: Try using alternative date columns: %s", other_date_cols)
                return None
        
        # Apply filters
        logger.info("Before filtering: rows=%d", len(working_df))
        for field, (operator, value) in filters.items():
            if field not in working_df.columns:
                continue
            if operator == "=":
                working_df = working_df[working_df[field].str.lower() == value.lower()]
            elif operator == ">":
                working_df = working_df[working_df[field] > float(value)]
            elif operator == "<":
                working_df = working_df[working_df[field] < float(value)]
            logger.info("After applying filter '%s %s %s': rows=%d", field, operator, value, len(working_df))
            logger.info("After applying filter '%s %s %s', unique values in %s: %s", field, operator, value, field, working_df[field].unique().tolist())
            if working_df.empty:
                logger.warning("DataFrame is empty after applying filter '%s %s %s'", field, operator, value)
                return None
        
        # Handle calculated metrics (e.g., "Profit / Sales")
        if metric and '/' in metric:
            try:
                working_df[metric] = evaluate_calculation(metric, working_df)
            except Exception as e:
                logger.error("Failed to evaluate calculated metric %s: %s", metric, str(e))
                return None
        
        # Handle trend over time
        if trend:
            dimension = dimension if dimension else trend
            if dimension == "Ship Month":
                working_df[dimension] = working_df[trend].dt.to_period('M').astype(str)
                chart_data = working_df.groupby([dimension])[metric].mean().reset_index()
            else:
                working_df['YearMonth'] = working_df[trend].dt.to_period('M')
                if dimension != trend:
                    chart_data = working_df.groupby(['YearMonth', dimension])[metric].mean().reset_index()
                else:
                    chart_data = working_df.groupby('YearMonth')[metric].mean().reset_index()
                chart_data['YearMonth'] = chart_data['YearMonth'].dt.to_timestamp()
                chart_data.rename(columns={'YearMonth': trend}, inplace=True)
            chart_type_default = "Line"
            if chart_data.empty:
                logger.error("Chart data is empty after grouping by %s", dimension)
                return None
        else:
            # Prioritize second_metric logic for Scatter plots over map_request
            if metric and second_metric:
                columns_to_include = [metric, second_metric]
                if dimension:
                    columns_to_include.append(dimension)
                chart_data = working_df[columns_to_include]
                chart_type_default = "Scatter"
            elif map_request and metric and dimension:
                logger.info("Processing map request: metric=%s, dimension=%s", metric, dimension)
                chart_data = working_df.groupby(dimension)[metric].mean().reset_index()
                if "highlight outliers" in prompt.lower():
                    chart_data = detect_outliers(chart_data, metric)
                    if chart_data is None:
                        logger.error("Failed to detect outliers for metric %s", metric)
                        return None
                    chart_data['Outlier_Label'] = chart_data['Is_Outlier'].apply(lambda x: 'Outlier' if x else 'Normal')
                    chart_type_default = "Map"
                else:
                    chart_type_default = "Map"
            elif dimension and metric:
                logger.info("Aggregating metric %s by dimension %s", metric, dimension)
                chart_data = working_df.groupby(dimension)[metric].mean().reset_index()
                chart_type_default = "Bar"
            elif metric:
                chart_data = pd.DataFrame({metric: [working_df[metric].mean()]})
                chart_type_default = "Bar"
            else:
                logger.error("Cannot generate chart: No valid metric or dimension identified.")
                return None
        
        if top_n:
            logger.info("Applying top %d filter on chart_data", top_n)
            chart_data = chart_data.sort_values(by=metric, ascending=False).head(top_n)
            # Check if the number of categories is less than requested top_n
            if len(chart_data) < top_n:
                logger.info("Requested top %d %s by %s, but only %d unique categories found", top_n, dimension, metric, len(chart_data))
        
        if chart_type_default not in ["Scatter", "Map"] and dimension and metric and sort_order and pd.api.types.is_numeric_dtype(chart_data[metric]):
            logger.info("Applied %s sorting on chart_data by %s: rows=%d", sort_order, metric, len(chart_data))
            chart_data = chart_data.sort_values(by=metric, ascending=(sort_order == "Ascending"))
        
        final_chart_type = chart_type if chart_type else chart_type_default
        
        if table_columns:
            chart_data = working_df[table_columns]
            final_chart_type = "Table"
        
        # Validate chart data for Bar and Line charts
        if final_chart_type in ["Bar", "Line"] and (dimension is None or dimension not in chart_data.columns):
            logger.error("Bar/Line chart requires a dimension, but none provided: dimension=%s", dimension)
            return None
        
        logger.info("Returning chart_data: rows=%d, columns=%s", len(chart_data), chart_data.columns.tolist())
        logger.info("Selected chart type for chart %d: %s (default was %s)", idx, final_chart_type, chart_type_default)
        return chart_data, metric, dimension, working_df, table_columns, final_chart_type
    
    except Exception as e:
        logger.error("Failed to render chart %d: %s", idx, str(e))
        return None