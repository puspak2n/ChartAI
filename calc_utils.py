import pandas as pd
import numpy as np
import re
import logging
import openai
import os

# Set up logging
logging.basicConfig(
    filename="chartgpt.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load OpenAI API key for formula generation
openai.api_key = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(openai.api_key)
if USE_OPENAI:
    logger.info("OpenAI API key loaded for formula generation.")
else:
    logger.warning("OpenAI API key not found for formula generation. Using rule-based generation.")

# Predefined calculations for common use cases
PREDEFINED_CALCULATIONS = {
    "Sales Indicator": {
        "prompt": "Mark Sales as High if greater than 1000, otherwise Low",
        "formula": "IF Sales > 1000 THEN 'High' ELSE 'Low' END"
    },
    "Profit Margin": {
        "prompt": "Calculate the profit margin as Profit divided by Sales",
        "formula": "Profit / Sales"
    },
    "Sales Outlier Flag": {
        "prompt": "Flag outliers in Sales where Sales is more than 2 standard deviations above the average",
        "formula": "IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END"
    },
    "Profit Above Average": {
        "prompt": "Calculate average Profit and flag if above overall average",
        "formula": "IF AVG(Profit) > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END"
    },
    "Profit Above Average by Segment": {
        "prompt": "Calculate average Profit per Segment and flag if above overall average",
        "formula": "IF AVG(Profit) PER Segment > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END"
    }
}

def generate_formula_with_gpt(prompt, dimensions, measures, df):
    """
    Generate a formula from a natural language prompt using OpenAI GPT.
    """
    if not USE_OPENAI:
        logger.warning("OpenAI API not available for formula generation. Falling back to rule-based parsing.")
        return None

    try:
        columns_str = ", ".join(df.columns)
        dimensions_str = ", ".join(dimensions)
        measures_str = ", ".join(measures)
        
        gpt_prompt = (
            f"Generate a formula based on the following natural language prompt: '{prompt}'. "
            f"Available columns: {columns_str}. Dimensions: {dimensions_str}. Measures: {measures_str}. "
            f"Use functions like SUM, AVG, COUNT, STDEV, MEDIAN, MIN, MAX, IF-THEN-ELSE-END. "
            f"Use 'PER' for grouped aggregations (e.g., AVG(Profit) PER Segment). "
            f"Do NOT use square brackets around column names (e.g., use Profit, not [Profit]). "
            f"Do NOT use 'OVER' for aggregations; use 'PER' instead. "
            f"Do NOT use 'ELSEIF'; use nested IF statements instead (e.g., IF condition1 THEN value1 ELSE IF condition2 THEN value2 ELSE value3 END END). "
            f"Ensure the formula uses exact column names from the dataset. "
            f"Examples: "
            f"- Prompt: 'Mark Sales as High if greater than 1000, otherwise Low' -> Formula: IF Sales > 1000 THEN 'High' ELSE 'Low' END "
            f"- Prompt: 'Calculate the profit margin as Profit divided by Sales' -> Formula: Profit / Sales "
            f"- Prompt: 'Flag outliers in Sales where Sales is more than 2 standard deviations above the average' -> Formula: IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END "
            f"- Prompt: 'Calculate average Profit per Ship Mode and flag if above overall average' -> Formula: IF AVG(Profit) PER Ship Mode > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END "
            f"- Prompt: 'If Sales is greater than 500 and Profit is positive, then High Performer, else if Sales is less than 200, then Low Performer, else Medium' -> Formula: IF Sales > 500 AND Profit > 0 THEN 'High Performer' ELSE IF Sales < 200 THEN 'Low Performer' ELSE 'Medium' END END "
            f"Return only the formula."
        )
        
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst skilled in generating formulas from natural language prompts."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=100,
            temperature=0.5
        )
        formula = response.choices[0].message.content.strip()
        logger.info("Generated GPT-based formula for prompt '%s': %s", prompt, formula)
        
        # Validate formula contains known columns, handling multi-word column names
        columns = set(df.columns)
        # Extract potential column names from the formula, preserving multi-word names
        used_columns = []
        # Match column names inside AVG() and STDEV()
        agg_matches = re.findall(r'(?:AVG|STDEV)\((.*?)\)', formula)
        used_columns.extend(agg_matches)
        # Match column names after PER
        per_match = re.search(r'PER\s+(.+?)(?=\s*(?:>|$))', formula)
        if per_match:
            used_columns.append(per_match.group(1).strip())
        # Match standalone column names (e.g., in IF conditions)
        condition_cols = re.findall(r'IF\s+(.+?)\s*(?:>|<|=|AND|OR)', formula)
        for cond in condition_cols:
            # Split by AND/OR and extract column names
            for part in re.split(r'\s+(?:AND|OR)\s+', cond):
                part = part.strip()
                # Match standalone column names
                col_match = re.match(r'(\w+(?:\s+\w+)*)', part)
                if col_match:
                    used_columns.append(col_match.group(1))
        # Match overall AVG/STDEV columns
        overall_cols = re.findall(r'AVG\((.*?)\)(?=\s*(?:THEN|$))', formula)
        used_columns.extend(overall_cols)
        
        for col in used_columns:
            col = col.strip()
            if col not in columns and col not in ['IF', 'THEN', 'ELSE', 'END', 'AVG', 'STDEV', 'SUM', 'COUNT', 'MEDIAN', 'MIN', 'MAX', 'PER', 'AND', 'OR']:
                logger.warning("Formula contains unknown column: %s", col)
                return None
        return formula
    except Exception as e:
        logger.error("Failed to generate GPT-based formula for prompt '%s': %s", prompt, str(e))
        return None

def generate_formula_from_prompt(prompt, dimensions, measures, df):
    """
    Generate a formula from a natural language prompt, using GPT if available.
    """
    try:
        prompt_lower = prompt.lower().strip()
        logger.info("Generating formula from prompt: %s", prompt)
        
        # Try GPT-based formula generation first
        gpt_formula = generate_formula_with_gpt(prompt, dimensions, measures, df)
        if gpt_formula:
            return gpt_formula
        
        # Check for predefined templates
        for template_name, template in PREDEFINED_CALCULATIONS.items():
            if template["prompt"].lower() in prompt_lower:
                # Special handling for "Profit Above Average by Segment" to replace Segment with the actual dimension
                if template_name == "Profit Above Average by Segment":
                    # Extract the dimension from the prompt (e.g., "per Ship Mode")
                    per_match = re.search(r'per\s+(.+?)(?=\s+and|$)', prompt_lower)
                    if per_match:
                        dim_part = per_match.group(1).strip()
                        # Find the matching dimension in the DataFrame columns
                        actual_dim = None
                        for dim in dimensions:
                            if dim.lower() == dim_part:
                                actual_dim = dim
                                break
                        if actual_dim:
                            formula = template["formula"].replace("Segment", actual_dim)
                            logger.info("Customized predefined template '%s' with dimension %s: %s", template_name, actual_dim, formula)
                            return formula
                        else:
                            logger.warning("Dimension '%s' not found in available dimensions", dim_part)
                            return None
                logger.info("Matched predefined template '%s': %s", template_name, template["formula"])
                return template["formula"]
        
        # Handle outlier prompts
        if "flag outliers in" in prompt_lower:
            outlier_match = re.search(r'flag outliers in (\w+)(\s+(?:by|per)\s+(\w+))? where (.+)', prompt_lower)
            if outlier_match:
                field = outlier_match.group(1)
                group_by = outlier_match.group(3)
                condition = outlier_match.group(4)
                for col in df.columns:
                    if col.lower() == field.lower():
                        field = col
                        break
                if group_by:
                    for col in df.columns:
                        if col.lower() == group_by.lower():
                            group_by = col
                            break
                if "standard deviations above the average" in condition:
                    std_dev_factor = re.search(r'(\d+)\s*standard deviations', condition)
                    factor = int(std_dev_factor.group(1)) if std_dev_factor else 2
                    formula = f"IF {field} > AVG({field}) + {factor} * STDEV({field}) THEN 'Outlier' ELSE 'Normal' END"
                    if group_by:
                        formula = f"IF {field} > AVG({field}) PER {group_by} + {factor} * STDEV({field}) PER {group_by} THEN 'Outlier' ELSE 'Normal' END"
                    logger.info("Generated formula from prompt '%s': %s", prompt, formula)
                    return formula
                elif "above the average" in condition:
                    formula = f"IF {field} > AVG({field}) THEN 'Outlier' ELSE 'Normal' END"
                    if group_by:
                        formula = f"IF {field} > AVG({field}) PER {group_by} THEN 'Outlier' ELSE 'Normal' END"
                    logger.info("Generated formula from prompt '%s': %s", prompt, formula)
                    return formula
        
        # Handle simple arithmetic
        if "calculate" in prompt_lower and "as" in prompt_lower:
            calc_match = re.search(r'calculate (.+?) as (.+)', prompt_lower)
            if calc_match:
                calc_name = calc_match.group(1).strip()
                expression = calc_match.group(2).strip()
                for measure in measures:
                    if measure.lower() in expression.lower():
                        expression = expression.replace(measure.lower(), measure)
                formula = expression
                logger.info("Generated formula from prompt '%s': %s", prompt, formula)
                return formula
        
        # Handle IF-THEN-ELSE conditions with nested IF for ELSEIF
        if "if" in prompt_lower and "then" in prompt_lower:
            if_match = re.search(r'if (.+?) then (.+?)( else if (.+?) then (.+?))? else (.+?) end', prompt_lower)
            if if_match:
                condition1 = if_match.group(1).strip()
                value1 = if_match.group(2).strip()
                condition2 = if_match.group(4).strip() if if_match.group(4) else None
                value2 = if_match.group(5).strip() if if_match.group(5) else None
                else_value = if_match.group(6).strip()
                for measure in measures:
                    if measure.lower() in condition1.lower():
                        condition1 = condition1.replace(measure.lower(), measure)
                    if condition2 and measure.lower() in condition2.lower():
                        condition2 = condition2.replace(measure.lower(), measure)
                    if measure.lower() in value1.lower():
                        value1 = value1.replace(measure.lower(), measure)
                    if value2 and measure.lower() in value2.lower():
                        value2 = value2.replace(measure.lower(), measure)
                    if measure.lower() in else_value.lower():
                        else_value = else_value.replace(measure.lower(), measure)
                if condition2 and value2:
                    formula = f"IF {condition1} THEN '{value1}' ELSE IF {condition2} THEN '{value2}' ELSE '{else_value}' END END"
                else:
                    formula = f"IF {condition1} THEN '{value1}' ELSE '{else_value}' END"
                logger.info("Generated formula from prompt '%s': %s", prompt, formula)
                return formula
        
        logger.warning("Could not generate formula from prompt: %s", prompt)
        return None
    except Exception as e:
        logger.error("Failed to generate formula from prompt '%s': %s", prompt, str(e))
        return None

def evaluate_calculation(formula, df, group_by=None):
    """
    Evaluate a formula on the DataFrame, supporting nested IF statements and grouped aggregations.
    """
    try:
        logger.info("Evaluating formula: %s", formula)
        
        working_df = df.copy()
        
        # Handle IF-THEN-ELSE logic, including nested IF statements
        if formula.startswith("IF"):
            # Parse nested IF statements iteratively
            def parse_if_formula(f, pos=0):
                conditions = []
                values = []
                else_value = None
                
                while pos < len(f):
                    if_match = re.match(r'IF\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+?))?\s+END', f[pos:], re.IGNORECASE)
                    if not if_match:
                        break
                    condition = if_match.group(1).strip()
                    value = if_match.group(2).strip("'")
                    else_part = if_match.group(3).strip("'") if if_match.group(3) else None
                    
                    conditions.append(condition)
                    values.append(value)
                    
                    # Check if the else part contains another IF
                    if else_part and else_part.lower().startswith('if'):
                        nested_conditions, nested_values, nested_else, new_pos = parse_if_formula(else_part, 3)  # Skip 'if '
                        conditions.extend(nested_conditions)
                        values.extend(nested_values)
                        if nested_else:
                            else_value = nested_else
                        pos += if_match.start(3) + new_pos + 4  # Move past the nested IF and 'END'
                    else:
                        else_value = else_part
                        pos += if_match.end()
                    if not else_part:
                        break
                
                return conditions, values, else_value, pos
            
            conditions, values, else_value, _ = parse_if_formula(formula)
            
            # Handle grouped aggregations (e.g., AVG(Profit) PER Ship Mode > AVG(Profit))
            if any("PER" in cond for cond in conditions):
                # Match pattern like: AVG(Profit) PER Ship Mode > AVG(Profit)
                agg_match = re.search(r'AVG\((\w+)\)\s+PER\s+(.+?)\s*>\s*AVG\((\w+)\)', conditions[0])
                if agg_match:
                    avg_field = agg_match.group(1)  # e.g., Profit (field for group AVG)
                    group_by = agg_match.group(2).strip()  # e.g., Ship Mode
                    overall_avg_field = agg_match.group(3)  # e.g., Profit (field for overall AVG)
                    
                    if avg_field not in working_df.columns:
                        raise ValueError(f"Field '{avg_field}' not found in DataFrame")
                    if overall_avg_field not in working_df.columns:
                        raise ValueError(f"Field '{overall_avg_field}' not found in DataFrame")
                    if group_by not in working_df.columns:
                        raise ValueError(f"Group-by field '{group_by}' not found in DataFrame")
                    
                    # Compute overall average
                    overall_avg = working_df[overall_avg_field].mean()
                    logger.info("Computed overall AVG(%s): %f", overall_avg_field, overall_avg)
                    
                    # Compute grouped mean
                    grouped_mean = working_df.groupby(group_by)[avg_field].transform('mean')
                    working_df[f'AVG_{avg_field}_PER_{group_by}'] = grouped_mean
                    logger.info("Computed AVG(%s) per %s: sample values=%s", avg_field, group_by, grouped_mean.head().tolist())
                    
                    # Compare each group's average to the overall average
                    group_means = working_df.groupby(group_by)[avg_field].mean().reset_index()
                    group_means['Above_Overall'] = group_means[avg_field] > overall_avg
                    group_means['Result'] = group_means['Above_Overall'].apply(lambda x: values[0] if x else else_value)
                    
                    # Merge the result back to the original DataFrame
                    result_df = working_df[[group_by]].merge(group_means[[group_by, 'Result']], on=group_by, how='left')
                    result = result_df['Result'].values
                    
                    logger.info("Computed grouped calculation for formula: %s, result sample=%s", formula, result[:5].tolist())
                    return result
            
            # Handle grouped aggregations with STDEV (e.g., Sales > AVG(Sales) PER Segment + 2 * STDEV(Sales) PER Segment)
            agg_match = re.search(r'(\w+)\s*>\s*AVG\((\w+)\)\s+PER\s+(\w+\s*\w*)\s*\+\s*(\d+)\s*\*\s*STDEV\((\w+)\)\s*(?:PER\s+(\w+\s*\w*))?', conditions[0])
            if agg_match:
                field = agg_match.group(1)  # e.g., Sales (the field to compare)
                avg_field = agg_match.group(2)  # e.g., Sales (field for AVG)
                group_by = agg_match.group(3).strip()  # e.g., Segment
                factor = int(agg_match.group(4))  # e.g., 2
                std_field = agg_match.group(5)  # e.g., Sales (field for STDEV)
                std_group_by = agg_match.group(6).strip() if agg_match.group(6) else group_by  # Should match group_by
                
                if field not in working_df.columns or avg_field not in working_df.columns:
                    raise ValueError(f"Field '{field}' or '{avg_field}' not found in DataFrame")
                if group_by not in working_df.columns:
                    raise ValueError(f"Group-by field '{group_by}' not found in DataFrame")
                if std_field not in working_df.columns:
                    raise ValueError(f"STDEV field '{std_field}' not found in DataFrame")
                if std_group_by != group_by:
                    raise ValueError(f"Group-by field for STDEV '{std_group_by}' does not match AVG group-by '{group_by}'")
                
                # Compute grouped mean
                grouped_mean = working_df.groupby(group_by)[avg_field].transform('mean')
                working_df[f'AVG_{avg_field}_PER_{group_by}'] = grouped_mean
                logger.info("Computed AVG(%s) per %s: sample values=%s", avg_field, group_by, grouped_mean.head().tolist())
                
                # Compute grouped standard deviation
                grouped_std = working_df.groupby(group_by)[std_field].transform('std')
                # Handle potential NaN values in std (e.g., if a group has only one value)
                grouped_std = grouped_std.fillna(0)
                working_df[f'STDEV_{std_field}_PER_{group_by}'] = grouped_std
                logger.info("Computed STDEV(%s) per %s: sample values=%s", std_field, group_by, grouped_std.head().tolist())
                
                # Compute the threshold per segment
                threshold = working_df[f'AVG_{avg_field}_PER_{group_by}'] + factor * working_df[f'STDEV_{std_field}_PER_{group_by}']
                logger.info("Computed threshold per %s: sample values=%s", group_by, threshold.head().tolist())
                
                # Compare each row's Sales against its Segment's threshold
                condition_result = working_df[field] > threshold
                logger.info("Condition result sample: %s", condition_result.head().tolist())
                
                result = pd.Series(np.where(condition_result, values[0], else_value), index=working_df.index)
                logger.info("Computed grouped calculation for formula: %s, result sample=%s", formula, result[:5].tolist())
                return result
            
            # Handle non-grouped IF conditions with nested IFs
            result = pd.Series([None] * len(working_df), index=working_df.index)  # Initialize as a Series
            for i, condition in enumerate(conditions):
                # Replace aggregate functions with computed values
                if "AVG(" in condition:
                    field_match = re.search(r'AVG\((.*?)\)', condition)
                    if field_match:
                        field = field_match.group(1)
                        if field not in working_df.columns:
                            raise KeyError(f"Column {field} not found in DataFrame")
                        avg_value = working_df[field].mean()
                        condition = condition.replace(f"AVG({field})", str(avg_value))
                
                if "STDEV(" in condition:
                    field_match = re.search(r'STDEV\((.*?)\)', condition)
                    if field_match:
                        field = field_match.group(1)
                        if field not in working_df.columns:
                            raise KeyError(f"Column {field} not found in DataFrame")
                        std_value = working_df[field].std()
                        condition = condition.replace(f"STDEV({field})", str(std_value))
                
                # Replace AND/OR with pandas operators and add parentheses
                condition = condition.replace("AND", "&").replace("OR", "|")
                # Add parentheses around comparisons to ensure correct precedence
                condition_parts = re.split(r'\s*(&|\|)\s*', condition)
                for j in range(len(condition_parts)):
                    part = condition_parts[j].strip()
                    if part not in ['&', '|']:
                        # Check if the part is a comparison (e.g., Sales > 500)
                        if any(op in part for op in ['>', '<', '=', '!=']):
                            condition_parts[j] = f"({part})"
                condition = ' '.join(condition_parts)
                
                # Replace column names with DataFrame references
                for col in working_df.columns:
                    if col in condition:
                        condition = condition.replace(col, f"working_df['{col}']")
                
                condition_result = eval(condition)
                # Update result: assign the value where condition is True and result is still None
                # Ensure result remains a pandas Series
                result = pd.Series(
                    np.where((condition_result) & (result.isna()), values[i], result),
                    index=working_df.index
                )
            
            # Apply the else value to remaining rows, keeping result as a Series
            result = pd.Series(
                np.where(result.isna(), else_value, result),
                index=working_df.index
            )
            return result
        
        # Handle simple arithmetic expressions
        working_formula = formula
        for col in working_df.columns:
            if col in working_formula:
                working_formula = working_formula.replace(col, f"working_df['{col}']")
        result = eval(working_formula)
        return result
    
    except Exception as e:
        logger.error("Failed to evaluate formula %s: %s", formula, str(e))
        raise

def detect_outliers(df, metric):
    """
    Detect outliers in a numeric column using IQR method.
    """
    try:
        if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
            logger.error("Metric %s is not numeric or not in DataFrame", metric)
            return None
        
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df['Is_Outlier'] = (df[metric] < lower_bound) | (df[metric] > upper_bound)
        logger.info("Detected outliers in %s: %d rows marked as outliers", metric, df['Is_Outlier'].sum())
        return df
    
    except Exception as e:
        logger.error("Failed to detect outliers in %s: %s", metric, str(e))
        return None

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a given column in the DataFrame.
    """
    if column not in df.columns:
        logger.error("Column %s not found in DataFrame for statistics calculation", column)
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