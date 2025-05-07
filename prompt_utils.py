import random
import pandas as pd
import openai
import logging
import os

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
        prompts = response.choices[0].message.content.strip().split('\n')
        prompts = [p.strip('- ').strip() for p in prompts if p.strip()]
        logger.info("Generated GPT-based sample prompts: %s", prompts)
        return prompts[:5]
    except Exception as e:
        logger.error("Failed to generate LLM-based prompts: %s", str(e))
        return None

def generate_sample_prompts(dimensions, measures, dates, df):
    """
    Generate a list of sample prompts using rule-based logic for varied and insightful prompts.
    """
    prioritized_dimensions, prioritized_measures, prioritized_dates = prioritize_fields(dimensions, measures, dates, df)
    
    if not prioritized_dimensions or not prioritized_measures:
        logger.warning("No prioritized dimensions or measures available for prompt generation.")
        return []

    prompts = []

    # Prompt 1: Top N ranking with a different dimension and measure
    if len(prioritized_dimensions) >= 2 and len(prioritized_measures) >= 1:
        dim = prioritized_dimensions[1]  # Use the second dimension to vary from Ship Mode
        measure = prioritized_measures[0]
        prompts.append(f"Top 3 {dim} by {measure}")

    # Prompt 2: Trend analysis with a different date and measure
    if prioritized_dates and len(prioritized_measures) >= 2:
        date = prioritized_dates[0]
        measure = prioritized_measures[1]  # Use a different measure
        prompts.append(f"{measure} trend over {date} by {prioritized_dimensions[0]}")

    # Prompt 3: Correlation between two measures
    if len(prioritized_measures) >= 2:
        prompts.append(f"Correlation between {prioritized_measures[0]} and {prioritized_measures[1]}")

    # Prompt 4: Filtered view with a different dimension and filter
    if len(prioritized_dimensions) >= 3:
        filter_dim = prioritized_dimensions[2]  # Use a different dimension for filtering
        unique_values = df[filter_dim].dropna().unique()
        if unique_values.size > 0:
            filter_value = unique_values[0]
            prompts.append(f"{prioritized_measures[0]} by {prioritized_dimensions[0]} with filter {filter_dim} = {filter_value}")

    # Prompt 5: Outlier detection with a different measure and dimension
    if len(prioritized_measures) >= 2 and len(prioritized_dimensions) >= 2:
        prompts.append(f"Find outliers in {prioritized_measures[1]} by {prioritized_dimensions[1]}")

    # Shuffle to ensure variety in display order
    random.shuffle(prompts)
    logger.info("Generated rule-based sample prompts: %s", prompts)
    return prompts[:5]
