# project_manager.py

import os
import pandas as pd

def list_projects():
    if not os.path.exists("projects"):
        os.makedirs("projects")
    return [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]

def load_project(project_name):
    try:
        prompts_df = pd.read_csv(f"projects/{project_name}/dashboard.csv")
        dataset_df = pd.read_csv(f"projects/{project_name}/dataset.csv")
        return prompts_df, dataset_df
    except Exception as e:
        print(f"Error loading project {project_name}: {e}")
        return None, None
