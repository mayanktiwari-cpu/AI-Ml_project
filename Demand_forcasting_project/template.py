import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "retail_inventory_ml"

list_of_files = [
    ".github/workflows/ci_cd.yml",
    "config/config.yaml",
    f"Demand_forcasting_project/{project_name}/__init__.py",
    f"Demand_forcasting_project/{project_name}/logger.py",
    f"Demand_forcasting_project/{project_name}/utils.py",
    f"Demand_forcasting_project/{project_name}/preprocessing.py",
    f"Demand_forcasting_project/{project_name}/feature_engineering.py",
    f"Demand_forcasting_project/{project_name}/train.py",
    f"Demand_forcasting_project/{project_name}/evaluate.py",
    f"Demand_forcasting_project/{project_name}/api.py",
    "tests/__init__.py",
    "tests/test_features.py",
    "tests/test_api.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "reports/figures/.gitkeep",
    "logs/.gitkeep",
    "Dockerfile",
    "requirements.txt",
    "setup_env.sh",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")