import random
import pandas as pd
import openai
import logging
import os
from utils import setup_logging

# Set up logging
logger = setup_logging()

# Load OpenAI API key for prompt generation
openai.api_key = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(openai.api_key)
if USE_OPENAI:
    logger.info("OpenAI API key loaded for prompt generation.")
else:
    logger.warning("OpenAI API key not found for prompt generation. Using rule-based generation.")

def prioritize_fields(dimensions, measures, dates, df):
    """
    Prioritize fields based on data quality and relevance.
    Returns: (prioritized_dimensions, prioritized_measures, prioritized_dates)
    """
    prioritized_dimensions = []
    prioritized_measures = []
    prioritized_dates = []

    for dim in dimensions:
        if dim in df.columns and df[dim].nunique() > 1 and df[dim].isna().mean() < 0.5:
            prioritized_dimensions.append(dim)

    for measure in measures:
        if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]) and df[measure].isna().mean() < 0.5:
            prioritized_measures.append(measure)

    for date in dates:
        if date in df.columns and df[date].notna().any():
            prioritized_dates.append(date)

    return prioritized_dimensions, prioritized_measures, prioritized_dates

def generate_sample_prompts(dimensions, measures, dates, df):
    """
    Generate a list of sample prompts based on allowed templates, dynamically using the dataset schema.
    Allowed templates:
    1. Trend over time: [Metric] by [Date]
    2. Trend by group over time: [Metric] by [Date] and [Dimension]
    3. Compare two metrics: [Metric1] vs [Metric2] by [Dimension]
    4. Top N: Top [N] [Dimension] by [Metric]
    5. Bottom N: Bottom [N] [Dimension] by [Metric]
    6. Top N with Filter: Top [N] [Dimension] by [Metric] where [Filter]
    7. Map chart: [Metric] by Country
    8. Outliers: Find outliers in [Metric] by [Dimension]
    9. Filter by category: [Metric] by [Dimension] where [Dimension = Value]
    10. Filter by value: [Metric] by [Dimension] where [Metric >= Value]
    """
    prioritized_dimensions, prioritized_measures, prioritized_dates = prioritize_fields(dimensions, measures, dates, df)
    
    if not prioritized_dimensions or not prioritized_measures or not prioritized_dates:
        logger.warning("No prioritized dimensions, measures, or dates available for prompt generation.")
        return []

    prompts = []

    # Ensure at least one prompt per allowed template (where applicable)
    # 1. Trend over time
    if prioritized_dates and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dates[0]}")

    # 2. Trend by group over time
    if prioritized_dates and prioritized_measures and prioritized_dimensions:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dates[0]} and {prioritized_dimensions[0]}")

    # 3. Compare two metrics
    if len(prioritized_measures) >= 2 and prioritized_dimensions:
        prompts.append(f"{prioritized_measures[0]} vs {prioritized_measures[1]} by {prioritized_dimensions[0]}")

    # 4. Top N
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Top 5 {prioritized_dimensions[0]} by {prioritized_measures[0]}")

    # 5. Bottom N
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Bottom 5 {prioritized_dimensions[1 % len(prioritized_dimensions)]} by {prioritized_measures[0]}")

    # 6. Top N with Filter
    if prioritized_dimensions and prioritized_measures and df[prioritized_dimensions[0]].nunique() > 0:
        filter_value = df[prioritized_dimensions[0]].dropna().unique()[0]
        prompts.append(f"Top 5 {prioritized_dimensions[0]} by {prioritized_measures[0]} where {prioritized_dimensions[0]} = {filter_value}")

    # 7. Map chart
    if "Country" in prioritized_dimensions and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by Country")

    # 8. Outliers
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Find outliers in {prioritized_measures[0]} by {prioritized_dimensions[0]}")

    # 9. Filter by category
    if prioritized_dimensions and prioritized_measures and df[prioritized_dimensions[0]].nunique() > 0:
        filter_value = df[prioritized_dimensions[0]].dropna().unique()[0]
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dimensions[0]} where {prioritized_dimensions[0]} = {filter_value}")

    # 10. Filter by value
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dimensions[0]} where {prioritized_measures[0]} >= 1000")

    # Shuffle and limit to 5 prompts to avoid overwhelming the UI
    random.shuffle(prompts)
    prompts = prompts[:5]
    numbered_prompts = [f"{i+1}. {p}" for i, p in enumerate(prompts)]
    logger.info("Generated sample prompts: %s", numbered_prompts)
    return numbered_prompts

def generate_prompts_with_llm(dimensions, measures, dates, df):
    """
    Generate sample prompts using OpenAI's GPT model.
    """
    if not USE_OPENAI:
        logger.info("Using rule-based prompt generation as per user preference.")
        return None

    try:
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

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst creating insightful visualization prompts for business users."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract prompts from the response
        content = response.choices[0].message.content
        prompts = [line.strip('- ').strip() for line in content.split('\n') if line.strip()]
        logger.info("Generated GPT-based sample prompts: %s", prompts)
        return prompts[:5]
    except Exception as e:
        logger.error("Failed to generate LLM-based prompts: %s", str(e))
        return None

def generate_sample_prompts(dimensions, measures, dates, df, max_prompts=5):
    """
    Generate a list of sample prompts based on allowed templates, dynamically using the dataset schema.
    """
    prioritized_dimensions, prioritized_measures, prioritized_dates = prioritize_fields(dimensions, measures, dates, df)
    
    if not prioritized_dimensions or not prioritized_measures:
        logger.warning("No prioritized dimensions or measures available for prompt generation.")
        return []

    prompts = []

    # 1. Trend over time
    if prioritized_dates and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dates[0]}")

    # 2. Trend by group
    if prioritized_dates and prioritized_measures and prioritized_dimensions:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dates[0]} and {prioritized_dimensions[0]}")

    # 3. Compare two metrics
    if len(prioritized_measures) >= 2 and prioritized_dimensions:
        prompts.append(f"{prioritized_measures[0]} vs {prioritized_measures[1]} by {prioritized_dimensions[0]}")

    # 4. Top N
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Top 5 {prioritized_dimensions[0]} by {prioritized_measures[0]}")

    # 5. Bottom N
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Bottom 5 {prioritized_dimensions[0]} by {prioritized_measures[0]}")

    # 6. Top N with filter
    if prioritized_dimensions and prioritized_measures and df[prioritized_dimensions[0]].nunique() > 0:
        filter_value = df[prioritized_dimensions[0]].dropna().unique()[0]
        prompts.append(f"Top 5 {prioritized_dimensions[0]} by {prioritized_measures[0]} where {prioritized_dimensions[0]} = {filter_value}")

    # 7. Map chart
    if "Country" in prioritized_dimensions and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by Country")

    # 8. Outliers
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"Find outliers in {prioritized_measures[0]} by {prioritized_dimensions[0]}")

    # 9. Filter by category
    if prioritized_dimensions and prioritized_measures and df[prioritized_dimensions[0]].nunique() > 0:
        filter_value = df[prioritized_dimensions[0]].dropna().unique()[0]
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dimensions[0]} where {prioritized_dimensions[0]} = {filter_value}")

    # 10. Filter by value
    if prioritized_dimensions and prioritized_measures:
        prompts.append(f"{prioritized_measures[0]} by {prioritized_dimensions[0]} where {prioritized_measures[0]} >= 1000")

    # Shuffle and limit
    random.shuffle(prompts)
    prompts = prompts[:max_prompts]
    numbered_prompts = [f"{i+1}. {p}" for i, p in enumerate(prompts)]
    logger.info("Generated sample prompts: %s", numbered_prompts)
    return numbered_prompts