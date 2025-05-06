# dashboard_manager.py

import json
import os

def save_dashboard(name, chart_history):
    save_path = os.path.join("dashboards", f"{name}.json")
    if not os.path.exists("dashboards"):
        os.makedirs("dashboards")
    serializable_history = []
    for item in chart_history:
        serializable_history.append({
            "prompt": item["prompt"],
            "data_summary": item["data_summary"],
        })
    with open(save_path, "w") as f:
        json.dump(serializable_history, f, indent=4)

def load_dashboard(file_name):
    load_path = os.path.join("dashboards", file_name)
    with open(load_path, "r") as f:
        loaded = json.load(f)

    loaded_history = []
    for item in loaded:
        loaded_history.append({
            "prompt": item["prompt"],
            "figure": None,  # Will need to rerun prompt to regenerate chart
            "data_summary": item["data_summary"]
        })
    return loaded_history
