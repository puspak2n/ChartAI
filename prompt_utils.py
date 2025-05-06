import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_sample_prompts(dimensions, measures, dates, df):
    """
    Generate sample prompts based on available dimensions, measures, and dates.
    Filters out inappropriate measures like Postal Code and IDs.
    """
    try:
        prompts = []
        
        # Filter out inappropriate measures (e.g., Postal Code, IDs)
        valid_measures = [m for m in measures if m.lower() != "postal code" and "id" not in m.lower()]
        if not valid_measures or not dimensions:
            logger.warning("No valid measures or dimensions for prompt generation: measures=%s, dimensions=%s", valid_measures, dimensions)
            return []
        
        # Prioritize business-relevant measures (e.g., Sales, Profit)
        priority_measures = ['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Calculate the profit margin as Profit divided by Sales']
        prioritized_measures = []
        for m in priority_measures:
            for vm in valid_measures:
                if vm.lower() == m.lower():
                    prioritized_measures.append(vm)
                    break
        # Add remaining measures that weren't in the priority list
        prioritized_measures.extend([m for m in valid_measures if m not in prioritized_measures])
        valid_measures = prioritized_measures
        
        if not valid_measures:
            logger.warning("No valid measures after prioritization")
            return []
        
        # Basic aggregation: Measure by Dimension
        prompts.append(f"{valid_measures[0]} by {dimensions[0]}")
        
        # Top N: Top 5 Dimension by Measure
        prompts.append(f"Top 5 {dimensions[0]} by {valid_measures[0]}")
        
        # Trend over time: Measure over Date
        if dates:
            prompts.append(f"{valid_measures[0]} trend over {dates[0]}")
        
        # Filtered prompt: Measure by Dimension with filter
        if len(dimensions) > 1:
            unique_values = df[dimensions[1]].dropna().unique()
            if len(unique_values) > 0:
                prompts.append(f"{valid_measures[0]} by {dimensions[0]} with filter {dimensions[1]} = {unique_values[0]}")
        
        # Comparison: Measure vs Measure by Dimension
        if len(valid_measures) > 1:
            prompts.append(f"{valid_measures[0]} vs {valid_measures[1]} by {dimensions[0]}")
        
        logger.info("Generated rule-based sample prompts: %s", prompts)
        return prompts
    except Exception as e:
        logger.error("Failed to generate rule-based sample prompts: %s", str(e))
        return []

def generate_prompts_with_llm(dimensions, measures, dates, df):
    """
    Use the rule-based method to generate sample prompts, as per user preference.
    """
    logger.info("Using rule-based prompt generation as per user preference.")
    return generate_sample_prompts(dimensions, measures, dates, df)

def prioritize_fields(df):
    """
    Prioritize fields for prompt generation based on their importance or usage.
    """
    try:
        dimensions = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])]
        measures = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        dates = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Prioritize measures based on variance
        prioritized_measures = []
        for measure in measures:
            if measure.lower() == "postal code" or "id" in measure.lower():
                continue
            variance = df[measure].var() if pd.api.types.is_numeric_dtype(df[measure]) else 0
            prioritized_measures.append((measure, variance))
        prioritized_measures.sort(key=lambda x: x[1], reverse=True)
        measures = [m[0] for m in prioritized_measures]
        
        # Prioritize dimensions based on cardinality
        prioritized_dimensions = []
        for dim in dimensions:
            cardinality = df[dim].nunique()
            prioritized_dimensions.append((dim, cardinality))
        prioritized_dimensions.sort(key=lambda x: x[1], reverse=True)
        dimensions = [d[0] for d in prioritized_dimensions]
        
        logger.info("Prioritized fields - Dimensions: %s, Measures: %s, Dates: %s", dimensions, measures, dates)
        return dimensions, measures, dates
    except Exception as e:
        logger.error("Failed to prioritize fields: %s", str(e))
        return [], [], []