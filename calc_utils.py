import pandas as pd
import numpy as np
import re
import logging
import os
import openai

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Predefined calculations for common patterns
PREDEFINED_CALCULATIONS = {
    "Sales Outlier Flag": {
        "prompt": "Flag outliers in Sales where Sales is more than 2 standard deviations above the average",
        "formula": "IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END"
    },
    "Profit Margin": {
        "prompt": "Calculate the profit margin as Profit divided by Sales",
        "formula": "Profit / Sales"
    },
    "Sales Threshold": {
        "prompt": "Mark Sales as High if greater than 1000, otherwise Low",
        "formula": "IF Sales > 1000 THEN 'High' ELSE 'Low' END"
    },
    "Profit Above Average by Ship Mode": {
        "prompt": "Calculate average Profit per Ship Mode and flag if above overall average",
        "formula": "IF AVG(Profit) PER Ship Mode > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END"
    }
}

def detect_outliers(df, metric, dimension=None):
    """
    Detect outliers in the specified metric, optionally grouped by dimension.
    Returns a DataFrame with an additional 'Outlier' column.
    """
    working_df = df.copy()
    logger.info("Starting outlier detection: metric=%s, dimension=%s, rows=%d", metric, dimension, len(working_df))

    # Validate metric column
    if metric not in working_df.columns:
        logger.error("Metric column '%s' not found in DataFrame", metric)
        working_df['Outlier'] = False
        return working_df

    # Ensure metric is numeric
    if not pd.api.types.is_numeric_dtype(working_df[metric]):
        logger.error("Metric column '%s' contains non-numeric data", metric)
        try:
            working_df[metric] = pd.to_numeric(working_df[metric], errors='coerce')
            logger.info("Converted metric '%s' to numeric, with NaN for non-numeric values", metric)
        except Exception as e:
            logger.error("Failed to convert metric '%s' to numeric: %s", metric, str(e))
            working_df['Outlier'] = False
            return working_df

    # Handle NaN values in metric
    if working_df[metric].isna().any():
        logger.warning("Metric column '%s' contains NaN values, filling with median", metric)
        working_df[metric] = working_df[metric].fillna(working_df[metric].median())

    try:
        if dimension:
            # Validate dimension column
            if dimension not in working_df.columns:
                logger.error("Dimension column '%s' not found in DataFrame", dimension)
                working_df['Outlier'] = False
                return working_df

            # Ensure dimension column is suitable for grouping
            if working_df[dimension].isna().all():
                logger.error("Dimension column '%s' contains all NaN values", dimension)
                working_df['Outlier'] = False
                return working_df

            # Drop rows where dimension is NaN to avoid grouping issues
            pre_group_rows = len(working_df)
            working_df = working_df.dropna(subset=[dimension])
            if len(working_df) < pre_group_rows:
                logger.warning("Dropped %d rows with NaN in dimension '%s'", pre_group_rows - len(working_df), dimension)

            # Group by dimension and calculate IQR for each group
            def calculate_outliers(group):
                if len(group) < 4:  # Need at least 4 values for meaningful quartiles
                    logger.warning("Group in dimension '%s' has too few values (%d), marking as non-outliers", dimension, len(group))
                    group['Outlier'] = False
                    return group
                # Check if all values in the group are NaN or constant
                if group[metric].isna().all() or group[metric].nunique() <= 1:
                    logger.warning("Group in dimension '%s' has all NaN or constant values, marking as non-outliers", dimension)
                    group['Outlier'] = False
                    return group
                Q1 = group[metric].quantile(0.25)
                Q3 = group[metric].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:  # Avoid division by zero or meaningless bounds
                    logger.warning("IQR is 0 for group in dimension '%s', marking as non-outliers", dimension)
                    group['Outlier'] = False
                    return group
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                group['Outlier'] = (group[metric] < lower_bound) | (group[metric] > upper_bound)
                return group

            # Apply outlier detection to each group
            result_df = working_df.groupby(dimension, group_keys=False).apply(calculate_outliers).reset_index(drop=True)
            logger.info("Completed outlier detection with dimension: rows=%d", len(result_df))
        else:
            # Check if all values are NaN or constant
            if working_df[metric].isna().all() or working_df[metric].nunique() <= 1:
                logger.warning("Metric '%s' has all NaN or constant values, marking as non-outliers", metric)
                working_df['Outlier'] = False
                result_df = working_df
            else:
                # Calculate IQR for the entire dataset
                Q1 = working_df[metric].quantile(0.25)
                Q3 = working_df[metric].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    logger.warning("IQR is 0 for metric '%s', marking as non-outliers", metric)
                    working_df['Outlier'] = False
                    result_df = working_df
                else:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    working_df['Outlier'] = (working_df[metric] < lower_bound) | (working_df[metric] > upper_bound)
                    result_df = working_df
            logger.info("Completed outlier detection without dimension: rows=%d", len(result_df))

        # Ensure 'Outlier' column exists and is boolean
        if 'Outlier' not in result_df.columns:
            logger.error("Outlier column not created, setting all to False")
            result_df['Outlier'] = False
        result_df['Outlier'] = result_df['Outlier'].astype(bool)

        return result_df

    except Exception as e:
        logger.error("Error in outlier detection: %s", str(e))
        working_df['Outlier'] = False
        return working_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a numeric column.
    """
    try:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None
        stats = {
            'mean': df[column].mean(),
            'std_dev': df[column].std(),
            'q1': df[column].quantile(0.25),
            'median': df[column].median(),
            'q3': df[column].quantile(0.75),
            'percentile_90': df[column].quantile(0.90)
        }
        return stats
    except Exception as e:
        logger.error("Failed to calculate statistics for %s: %s", column, str(e))
        return None

def generate_formula_with_gpt(prompt, dimensions, measures, df):
    """
    Generate a formula using OpenAI GPT based on the prompt, ensuring the correct IF-THEN-ELSE-END syntax.
    """
    try:
        columns = list(df.columns)
        system_prompt = (
            "You are a data analyst tasked with generating a formula based on a user's natural language prompt. "
            "The formula must use the following functions: SUM, AVG, COUNT, STDEV, MEDIAN, MIN, MAX, IF-THEN-ELSE-END. "
            "Use exact column names from the dataset. Available columns: " + ", ".join(columns) + ". "
            "Ensure the formula follows the syntax 'IF condition THEN value1 ELSE value2 END' for conditional logic, "
            "and use 'PER' for group-by aggregations (e.g., AVG(Profit) PER Ship Mode). "
            "For nested conditions, use multiple IF-THEN-ELSE-END statements."
        )
        response = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a formula for: {prompt}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        formula = response.choices[0].message.content.strip()
        # Ensure the formula ends with END if it contains IF-THEN-ELSE
        if "IF" in formula.upper() and "THEN" in formula.upper() and "ELSE" in formula.upper() and not formula.upper().endswith("END"):
            formula += " END"
        logger.info("Generated GPT-based formula: %s", formula)
        return formula
    except Exception as e:
        logger.error("Failed to generate GPT-based formula for prompt '%s': %s", prompt, str(e))
        return None

def generate_formula_from_prompt(prompt, dimensions, measures, df):
    """
    Generate a formula from a natural language prompt, either via GPT or predefined templates.
    """
    logger.info("Generating formula from prompt: \"%s\"", prompt)
    
    # First, try GPT if API key is available
    if os.getenv("OPENAI_API_KEY"):
        formula = generate_formula_with_gpt(prompt, dimensions, measures, df)
        if formula:
            return formula
    
    # If GPT fails or no API key, fall back to predefined templates
    for template_name, template in PREDEFINED_CALCULATIONS.items():
        if template["prompt"].lower() in prompt.lower() or prompt.lower() in template["prompt"].lower():
            logger.info("Matched predefined template '%s': %s", template_name, template["formula"])
            return template["formula"]
    
    # If no match is found, log a warning
    logger.warning("Could not generate formula from prompt: \"%s\"", prompt)
    return None

def evaluate_calculation(formula, df, group_by=None):
    """
    Evaluate a formula on the DataFrame, supporting aggregations, conditionals, and nested IF statements.
    """
    logger.info("Evaluating formula: %s", formula)
    
    try:
        # Replace aggregate functions with computed values
        working_df = df.copy()
        
        # Handle PER (group-by) aggregations
        if group_by and " PER " in formula.upper():
            agg_matches = re.findall(r'(AVG|SUM|COUNT|STDEV|MEDIAN|MIN|MAX)\((\w+)\)\s+PER\s+(\w+)', formula.upper())
            for agg_func, col, group_col in agg_matches:
                if col in working_df.columns and group_col in working_df.columns:
                    if agg_func == "AVG":
                        grouped = working_df.groupby(group_col)[col].mean()
                    elif agg_func == "SUM":
                        grouped = working_df.groupby(group_col)[col].sum()
                    elif agg_func == "COUNT":
                        grouped = working_df.groupby(group_col)[col].count()
                    elif agg_func == "STDEV":
                        grouped = working_df.groupby(group_col)[col].std()
                    elif agg_func == "MEDIAN":
                        grouped = working_df.groupby(group_col)[col].median()
                    elif agg_func == "MIN":
                        grouped = working_df.groupby(group_col)[col].min()
                    elif agg_func == "MAX":
                        grouped = working_df.groupby(group_col)[col].max()
                    else:
                        continue
                    # Map the grouped values back to the original DataFrame
                    working_df[f"{agg_func}({col})_PER_{group_col}"] = working_df[group_col].map(grouped)
                    formula = formula.replace(f"{agg_func}({col}) PER {group_col}", f"{agg_func}({col})_PER_{group_col}")
        
        # Compute overall aggregates
        agg_matches = re.findall(r'(AVG|SUM|COUNT|STDEV|MEDIAN|MIN|MAX)\((\w+)\)', formula.upper())
        for agg_func, col in agg_matches:
            if col in working_df.columns:
                if agg_func == "AVG":
                    value = working_df[col].mean()
                elif agg_func == "SUM":
                    value = working_df[col].sum()
                elif agg_func == "COUNT":
                    value = working_df[col].count()
                elif agg_func == "STDEV":
                    value = working_df[col].std()
                elif agg_func == "MEDIAN":
                    value = working_df[col].median()
                elif agg_func == "MIN":
                    value = working_df[col].min()
                elif agg_func == "MAX":
                    value = working_df[col].max()
                else:
                    continue
                formula = formula.replace(f"{agg_func}({col})", str(value))
        
        # Evaluate IF-THEN-ELSE-END expressions, including nested IFs
        def evaluate_condition(condition, df_row):
            condition = condition.replace(" AND ", " & ").replace(" OR ", " | ")
            for col in df.columns:
                if col in condition:
                    condition = condition.replace(col, f"df_row['{col}']")
            return eval(condition)
        
        def parse_if_statement(formula_str):
            # Match the outermost IF-THEN-ELSE-END
            pattern = r'IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END'
            match = re.match(pattern, formula_str, re.IGNORECASE)
            if not match:
                # If no IF-THEN-ELSE-END, treat as an arithmetic expression
                for col in working_df.columns:
                    if col in formula_str:
                        formula_str = formula_str.replace(col, f"working_df['{col}']")
                return eval(formula_str)
            
            condition = match.group(1)
            then_value = match.group(2)
            else_value = match.group(3)

            # Check for nested IF in then_value or else_value
            if "IF" in then_value.upper():
                then_value = parse_if_statement(then_value)
            else:
                # Handle quoted strings or evaluate as expression
                then_value = then_value.strip("'").strip('"') if (then_value.startswith("'") or then_value.startswith('"')) else eval(then_value, {"working_df": working_df})

            if "IF" in else_value.upper():
                else_value = parse_if_statement(else_value)
            else:
                else_value = else_value.strip("'").strip('"') if (else_value.startswith("'") or else_value.startswith('"')) else eval(else_value, {"working_df": working_df})

            # Apply the condition row-wise
            result = pd.Series([
                then_value if evaluate_condition(condition, working_df.iloc[[i]]) else else_value
                for i in range(len(working_df))
            ], index=working_df.index)
            return result
        
        # Start parsing the formula
        result = parse_if_statement(formula)
        return result if isinstance(result, pd.Series) else pd.Series(result, index=working_df.index)
    
    except Exception as e:
        logger.error("Failed to evaluate formula '%s': %s", formula, str(e))
        # Fallback to predefined template if evaluation fails
        for template_name, template in PREDEFINED_CALCULATIONS.items():
            if template["prompt"].lower() in formula.lower():
                logger.info("Falling back to predefined template '%s': %s", template_name, template["formula"])
                return evaluate_calculation(template["formula"], df, group_by)
        return None
