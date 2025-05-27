import pandas as pd
import numpy as np
import logging
import re
from utils import setup_logging

# Set up logging
logger = setup_logging()

PREDEFINED_CALCULATIONS = {
    "Profit Margin": {
        "prompt": "Calculate the profit margin as Profit divided by Sales",
        "formula": "Profit / Sales"
    },
    "High Sales Flag": {
        "prompt": "Mark Sales as High if greater than 1000, otherwise Low",
        "formula": "IF Sales > 1000 THEN 'High' ELSE 'Low' END"
    },
    "Outlier Flag": {
        "prompt": "Flag outliers in Sales where Sales is more than 2 standard deviations above the average",
        "formula": "IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END"
    }
}

def evaluate_calculation(formula, df):
    """
    Evaluate a formula on a DataFrame and return the result.
    """
    try:
        # Placeholder for actual formula evaluation logic
        # This would involve parsing the formula and applying it to df
        logger.info("Evaluating formula: %s", formula)
        return df.apply(lambda row: 0, axis=1)  # Dummy implementation
    except Exception as e:
        logger.error("Failed to evaluate formula %s: %s", formula, str(e))
        return None

def generate_formula_from_prompt(prompt, dimensions, measures, df):
    """
    Generate a formula from a natural language prompt.
    """
    try:
        prompt_lower = prompt.lower()
        logger.info("Generating formula from prompt: %s", prompt)

        # Check for predefined calculations
        for calc_name, calc_info in PREDEFINED_CALCULATIONS.items():
            if calc_info["prompt"].lower() in prompt_lower:
                return calc_info["formula"]

        # Basic parsing for IF-THEN-ELSE statements
        if "if" in prompt_lower and "then" in prompt_lower:
            # Example: "Mark Sales as High if greater than 1000, otherwise Low"
            match = re.search(r'if (.+?) then (.+?) else (.+)', prompt_lower)
            if match:
                condition = match.group(1)
                then_value = match.group(2)
                else_value = match.group(3)
                # Translate to formula syntax
                condition = condition.replace("greater than", ">").replace("less than", "<")
                for measure in measures:
                    condition = condition.replace(measure.lower(), measure)
                formula = f"IF {condition} THEN '{then_value}' ELSE '{else_value}' END"
                return formula

        # Basic parsing for arithmetic operations
        if "calculate" in prompt_lower and "as" in prompt_lower:
            # Example: "Calculate the profit margin as Profit divided by Sales"
            match = re.search(r'calculate .+? as (.+)', prompt_lower)
            if match:
                expression = match.group(1)
                for measure in measures:
                    expression = expression.replace(measure.lower(), measure)
                expression = expression.replace("divided by", "/").replace("multiplied by", "*")
                return expression

        logger.warning("Could not generate formula from prompt: %s", prompt)
        return None
    except Exception as e:
        logger.error("Failed to generate formula from prompt %s: %s", prompt, str(e))
        return None

def detect_outliers(df, metric, dimension=None, method="std", threshold=1):
    """
    Detect outliers using various methods:
    - std: Standard deviation (default, uses threshold=1)
    - iqr: Interquartile range
    - zscore: Z-score method
    - mad: Median Absolute Deviation
    
    Returns a DataFrame with an 'Outlier' column (boolean).
    """
    try:
        # Calculate outliers on the entire dataset first
        if method == "std":
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            upper_bound = mean_val + threshold * std_val
            lower_bound = mean_val - threshold * std_val
        elif method == "iqr":
            Q1 = df[metric].quantile(0.25)
            Q3 = df[metric].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
        elif method == "zscore":
            z_scores = np.abs((df[metric] - df[metric].mean()) / df[metric].std())
            df['Outlier'] = z_scores > threshold
            return df
        elif method == "mad":
            median_val = df[metric].median()
            mad = np.median(np.abs(df[metric] - median_val))
            upper_bound = median_val + threshold * mad
            lower_bound = median_val - threshold * mad
        
        # Mark outliers in the original data
        df['Outlier'] = (df[metric] > upper_bound) | (df[metric] < lower_bound)
        
        # If dimension is provided, aggregate the outlier information
        if dimension and dimension in df.columns:
            # Group by dimension and check if any value in the group is an outlier
            grouped_outliers = df.groupby(dimension)['Outlier'].any().reset_index()
            df = df.merge(grouped_outliers, on=dimension, suffixes=('', '_group'))
            df['Outlier'] = df['Outlier_group']
            df = df.drop(columns=['Outlier_group'])
        
        logger.info(f"Detected outliers in {metric} using {method} method: upper_bound={upper_bound:.2f}, lower_bound={lower_bound:.2f}")
        
        # Log outlier statistics
        outlier_count = df['Outlier'].value_counts()
        logger.info(f"Outlier detection results: {outlier_count.to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error detecting outliers for metric {metric}: {str(e)}")
        raise

def generate_formula_from_prompt(prompt, dimensions, measures, df):
    """
    Generate a formula from a natural language prompt.
    Returns a string formula compatible with evaluate_calculation.
    """
    prompt_lower = prompt.lower().strip()
    logger.info("Generating formula from prompt: %s", prompt)

    try:
        # Handle simple arithmetic (e.g., "Profit Margin as Profit divided by Sales")
        if "divided by" in prompt_lower:
            parts = prompt_lower.split("divided by")
            if len(parts) == 2:
                numerator = parts[0].strip()
                denominator = parts[1].strip()
                for measure in measures:
                    if measure.lower() in numerator:
                        numerator = measure
                    if measure.lower() in denominator:
                        denominator = measure
                formula = f"{numerator} / {denominator}"
                logger.info("Generated arithmetic formula: %s", formula)
                return formula

        # Handle IF statements (e.g., "Mark Sales as High if greater than 1000, otherwise Low")
        if "if " in prompt_lower and ("then " in prompt_lower or "else " in prompt_lower):
            # Already handled by parse_if_statement in main.py
            formula = prompt
            logger.info("Detected IF statement, passing to parse_if_statement: %s", formula)
            return formula

        # Handle statistical conditions (e.g., "Flag outliers in Sales where Sales is more than 2 standard deviations above the average")
        if "more than" in prompt_lower and "standard deviations" in prompt_lower:
            metric_match = None
            for measure in measures:
                if measure.lower() in prompt_lower:
                    metric_match = measure
                    break
            if not metric_match:
                logger.warning("No metric found in prompt: %s", prompt_lower)
                return None
            
            # Calculate mean and std dev
            mean_val = df[metric_match].mean()
            std_val = df[metric_match].std()
            
            std_dev_match = re.search(r'(\d+)\s*standard deviations', prompt_lower)
            if std_dev_match:
                num_std = int(std_dev_match.group(1))
                threshold = mean_val + num_std * std_val
                formula = f"IF {metric_match} > {threshold} THEN 'Outlier' ELSE 'Normal' END"
                logger.info("Generated outlier formula: %s", formula)
                return formula

        # Handle group-wise averages (e.g., "Calculate average Sales per Ship Mode and flag if above overall average")
        if "average" in prompt_lower and "per" in prompt_lower and "flag if" in prompt_lower:
            metric_match = None
            for measure in measures:
                if measure.lower() in prompt_lower:
                    metric_match = measure
                    break
            if not metric_match:
                logger.warning("No metric found in prompt: %s", prompt_lower)
                return None

            dim_match = None
            for dim in dimensions:
                if dim.lower() in prompt_lower:
                    dim_match = dim
                    break
            if not dim_match:
                logger.warning("No dimension found in prompt: %s", prompt_lower)
                return None

            overall_avg = df[metric_match].mean()
            formula = f"IF AVG({metric_match}) PER {dim_match} > {overall_avg} THEN 'Above Average' ELSE 'Below Average' END"
            logger.info("Generated group-wise average formula: %s", formula)
            return formula

        logger.warning("Could not generate formula from prompt: %s", prompt_lower)
        return None
    except Exception as e:
        logger.error("Error generating formula from prompt '%s': %s", prompt_lower, str(e))
        return None

def calculate_statistics(df, metric):
    """
    Calculate basic statistics for a metric in a DataFrame.
    """
    try:
        if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
            logger.error("Metric %s not found or not numeric in DataFrame", metric)
            return None

        stats = {
            "mean": float(df[metric].mean()),
            "std_dev": float(df[metric].std()),
            "q1": float(df[metric].quantile(0.25)),
            "median": float(df[metric].median()),
            "q3": float(df[metric].quantile(0.75)),
            "percentile_90": float(df[metric].quantile(0.90)),
            "min": float(df[metric].min()),
            "max": float(df[metric].max())
        }
        return stats
    except Exception as e:
        logger.error("Failed to calculate statistics for metric %s: %s", metric, str(e))
        return None